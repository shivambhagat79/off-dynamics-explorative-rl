import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """

    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)

        return action * self.max_action, logprob, mean * self.max_action


class DoubleQFunc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


# domain classifier for DARC
class Classifier(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256, gaussian_noise_std=1.0):
        super(Classifier, self).__init__()
        self.action_dim = action_dim
        self.gaussian_noise_std = gaussian_noise_std
        self.sa_classifier = MLPNetwork(state_dim + action_dim, 2, hidden_size)
        self.sas_classifier = MLPNetwork(2 * state_dim + action_dim, 2, hidden_size)

    def forward(self, state_batch, action_batch, nextstate_batch, with_noise):
        sas = torch.cat([state_batch, action_batch, nextstate_batch], -1)

        if with_noise:
            sas += (
                torch.randn_like(sas, device=state_batch.device)
                * self.gaussian_noise_std
            )
        sas_logits = torch.nn.Softmax(dim=-1)(self.sas_classifier(sas))

        sa = torch.cat([state_batch, action_batch], -1)

        if with_noise:
            sa += (
                torch.randn_like(sa, device=state_batch.device)
                * self.gaussian_noise_std
            )
        sa_logits = torch.nn.Softmax(dim=-1)(self.sa_classifier(sa))

        return sas_logits, sa_logits


class TUNE_SE(object):
    def __init__(
        self,
        config,
        device,
        target_entropy=None,
    ):
        self.config = config
        self.device = device
        self.discount = config["gamma"]
        self.tau = config["tau"]
        self.target_entropy = (
            target_entropy if target_entropy else -config["action_dim"]
        )
        self.update_interval = config["update_interval"]

        self.total_it = 0

        # aka critic
        self.q_funcs = DoubleQFunc(
            config["state_dim"],
            config["action_dim"],
            hidden_size=config["hidden_sizes"],
        ).to(self.device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(
            config["state_dim"],
            config["action_dim"],
            config["max_action"],
            hidden_size=config["hidden_sizes"],
        ).to(self.device)

        # aka temperature
        if config["temperature_opt"]:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([0.2])).to(self.device)

        # aka classifier
        self.classifier = Classifier(
            config["state_dim"],
            config["action_dim"],
            config["hidden_sizes"],
            config["gaussian_noise_std"],
        ).to(self.device)

        self.q_optimizer = torch.optim.Adam(
            self.q_funcs.parameters(), lr=config["critic_lr"]
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config["actor_lr"]
        )
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=config["actor_lr"])
        self.classifier_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=config["actor_lr"]
        )

        # Source model ensemble for fine tuning (dynamics models: (s, a) -> s')
        self.tuning_ensemble = []
        self.tuning_ensemble_optimizer = []
        for _ in range(config["tuning_ensemble_size"]):
            network = MLPNetwork(
                config["state_dim"] + config["action_dim"], config["state_dim"]
            ).to(self.device)
            optimizer = torch.optim.Adam(network.parameters(), lr=config["tuning_lr"])

            self.tuning_ensemble.append(network)
            self.tuning_ensemble_optimizer.append(optimizer)

        # Running statistics for intrinsic reward normalization (RND-style).
        # Uses Welford's online algorithm to track mean/variance across all
        # batches of intrinsic rewards seen so far, ensuring the bonus stays
        # on a stable scale regardless of state dimensionality or training stage.
        self.intrinsic_reward_running_mean = 0.0
        self.intrinsic_reward_running_var = 1.0
        self.intrinsic_reward_count = 1e-4  # small eps to avoid div-by-zero initially

    def _update_intrinsic_reward_stats(self, batch):
        """Update running mean/variance with a new batch using parallel Welford merge."""
        batch_mean = batch.mean().item()
        batch_var = batch.var().item() if batch.numel() > 1 else 0.0
        batch_count = batch.numel()

        delta = batch_mean - self.intrinsic_reward_running_mean
        total_count = self.intrinsic_reward_count + batch_count

        new_mean = (
            self.intrinsic_reward_running_mean + delta * batch_count / total_count
        )
        m_a = self.intrinsic_reward_running_var * self.intrinsic_reward_count
        m_b = batch_var * batch_count
        m2 = (
            m_a
            + m_b
            + delta**2 * self.intrinsic_reward_count * batch_count / total_count
        )

        self.intrinsic_reward_running_mean = new_mean
        self.intrinsic_reward_running_var = m2 / total_count
        self.intrinsic_reward_count = total_count

    def select_action(self, state, test=True):
        with torch.no_grad():
            action, _, mean = self.policy(
                torch.Tensor(state).view(1, -1).to(self.device)
            )
        if test:
            return mean.squeeze().cpu().numpy()
        else:
            return action.squeeze().cpu().numpy()

    def select_tuned_action(self, state, tar_replay_buffer, batch_size, tune_steps=10):
        """
        Clone the current policy and critic, fine-tune the clones on the target
        replay buffer with intrinsic rewards derived from the source dynamics
        ensemble, select an action with the tuned policy, then discard all
        cloned components.

        The intrinsic reward is the L2 distance between the actual target
        next_state and the mean prediction of the source dynamics ensemble.
        Transitions where the target dynamics diverge most from the learned
        source dynamics receive the highest bonus, encouraging the tuned policy
        to explore regions unique to the target domain.

        Args:
            state: Current state (numpy array).
            tar_replay_buffer: Replay buffer containing target domain transitions.
            batch_size: Batch size for sampling from the target buffer.
            tune_steps: Number of SAC gradient steps to fine-tune the clone.

        Returns:
            action: Action selected by the temporarily tuned policy (numpy array).
        """
        # If the target buffer doesn't have enough data, fall back to the original policy
        if tar_replay_buffer.size < batch_size:
            return self.select_action(state, test=False)

        # --- 1. Clone policy, critic, target critic, and temperature ---
        tuned_policy = copy.deepcopy(self.policy)
        tuned_q_funcs = copy.deepcopy(self.q_funcs)
        tuned_target_q_funcs = copy.deepcopy(self.target_q_funcs)
        tuned_target_q_funcs.eval()
        for p in tuned_target_q_funcs.parameters():
            p.requires_grad = False

        if self.config["temperature_opt"]:
            tuned_log_alpha = self.log_alpha.clone().detach().requires_grad_(True)
        else:
            tuned_log_alpha = self.log_alpha.clone().detach()

        # Create optimizers for the cloned components
        tuned_q_optimizer = torch.optim.Adam(
            tuned_q_funcs.parameters(), lr=self.config["critic_lr"]
        )
        tuned_policy_optimizer = torch.optim.Adam(
            tuned_policy.parameters(), lr=self.config["actor_lr"]
        )
        tuned_temp_optimizer = None
        if self.config["temperature_opt"]:
            tuned_temp_optimizer = torch.optim.Adam(
                [tuned_log_alpha], lr=self.config["actor_lr"]
            )

        # --- 2. Fine-tune the cloned networks on target buffer data ---
        for _ in range(tune_steps):
            tar_state, tar_action, tar_next_state, tar_reward, tar_not_done = (
                tar_replay_buffer.sample(batch_size)
            )

            # -- Compute intrinsic reward from source dynamics ensemble --
            # Each ensemble member was trained to predict s' under SOURCE dynamics.
            # When fed target (s, a), the gap between predicted (source) s' and
            # actual (target) s' measures how much the dynamics differ for this
            # transition. A large gap means this region is unique to the target
            # domain, so we reward the agent for visiting it.
            with torch.no_grad():
                model_input = torch.cat([tar_state, tar_action], dim=1)
                ensemble_predictions = torch.stack(
                    [network(model_input) for network in self.tuning_ensemble],
                    dim=0,
                )  # (ensemble_size, batch_size, state_dim)
                mean_prediction = ensemble_predictions.mean(
                    dim=0
                )  # (batch_size, state_dim)

                # L2 norm of the difference: scalar per sample
                intrinsic_reward = torch.norm(
                    tar_next_state - mean_prediction, p=2, dim=1, keepdim=True
                )  # (batch_size, 1)

            # Update running statistics BEFORE normalizing (outside no_grad
            # since this only touches Python scalars, but intrinsic_reward
            # itself is detached/no-grad already).
            self._update_intrinsic_reward_stats(intrinsic_reward)

            # Normalize using running statistics (RND-style): subtract the
            # running mean so that only above-average dynamics gaps yield a
            # positive bonus, then divide by running std to keep the scale
            # consistent across training.
            with torch.no_grad():
                running_std = max(self.intrinsic_reward_running_var, 1e-8) ** 0.5
                intrinsic_reward = (
                    intrinsic_reward - self.intrinsic_reward_running_mean
                ) / running_std

                # Clamp to [0, 5]: floor at 0 so below-average gaps never
                # penalise, ceiling at 5 to prevent reward explosion from
                # rare extreme outliers.
                intrinsic_reward = torch.clamp(intrinsic_reward, min=0.0, max=5.0)

                augmented_reward = tar_reward + intrinsic_reward

            # -- Update Q-functions (critic) --
            with torch.no_grad():
                next_action, logprobs, _ = tuned_policy(
                    tar_next_state, get_logprob=True
                )
                q_t1, q_t2 = tuned_target_q_funcs(tar_next_state, next_action)
                q_target = torch.min(q_t1, q_t2)
                tuned_alpha = tuned_log_alpha.exp()
                value_target = augmented_reward + tar_not_done * self.discount * (
                    q_target - tuned_alpha * logprobs
                )

            q_1, q_2 = tuned_q_funcs(tar_state, tar_action)
            q_loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)

            tuned_q_optimizer.zero_grad()
            q_loss.backward()
            tuned_q_optimizer.step()

            # -- Update target Q-functions via Polyak averaging --
            with torch.no_grad():
                for target_param, param in zip(
                    tuned_target_q_funcs.parameters(), tuned_q_funcs.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1.0 - self.tau) * target_param.data
                    )

            # -- Update policy (actor) --
            for p in tuned_q_funcs.parameters():
                p.requires_grad = False

            action_batch, logprobs_batch, _ = tuned_policy(tar_state, get_logprob=True)
            q_b1, q_b2 = tuned_q_funcs(tar_state, action_batch)
            qval_batch = torch.min(q_b1, q_b2)
            tuned_alpha = tuned_log_alpha.exp()
            policy_loss = (tuned_alpha * logprobs_batch - qval_batch).mean()

            tuned_policy_optimizer.zero_grad()
            policy_loss.backward()
            tuned_policy_optimizer.step()

            # -- Update temperature --
            if self.config["temperature_opt"] and tuned_temp_optimizer is not None:
                temp_loss = (
                    -tuned_alpha
                    * (logprobs_batch.detach() + self.target_entropy).mean()
                )
                tuned_temp_optimizer.zero_grad()
                temp_loss.backward()
                tuned_temp_optimizer.step()

            for p in tuned_q_funcs.parameters():
                p.requires_grad = True

        # --- 3. Select action using the tuned policy ---
        with torch.no_grad():
            action, _, _ = tuned_policy(torch.Tensor(state).view(1, -1).to(self.device))
        action = action.squeeze().cpu().numpy()

        # --- 4. Discard all cloned components ---
        del tuned_policy, tuned_q_funcs, tuned_target_q_funcs
        del tuned_q_optimizer, tuned_policy_optimizer, tuned_log_alpha
        if tuned_temp_optimizer is not None:
            del tuned_temp_optimizer

        return action

    def update_classifier(
        self, src_replay_buffer, tar_replay_buffer, batch_size, writer=None
    ):
        src_state, src_action, src_next_state, _, _ = src_replay_buffer.sample(
            batch_size
        )
        tar_state, tar_action, tar_next_state, _, _ = tar_replay_buffer.sample(
            batch_size
        )

        state = torch.cat([src_state, tar_state], 0)
        action = torch.cat([src_action, tar_action], 0)
        next_state = torch.cat([src_next_state, tar_next_state], 0)

        # set labels for different domains
        label = (
            torch.cat(
                [torch.zeros(size=(batch_size,)), torch.ones(size=(batch_size,))], dim=0
            )
            .long()
            .to(self.device)
        )

        indexs = torch.randperm(label.shape[0])
        state_batch, action_batch, nextstate_batch = (
            state[indexs],
            action[indexs],
            next_state[indexs],
        )
        label = label[indexs]

        sas_logits, sa_logits = self.classifier(
            state_batch, action_batch, nextstate_batch, with_noise=True
        )
        loss_sas = F.cross_entropy(sas_logits, label)
        loss_sa = F.cross_entropy(sa_logits, label)
        classifier_loss = loss_sas + loss_sa
        self.classifier_optimizer.zero_grad()
        classifier_loss.backward()
        self.classifier_optimizer.step()

        # log necessary information if the logger is not None
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar(
                "train/sas classifier loss", loss_sas, global_step=self.total_it
            )
            writer.add_scalar(
                "train/sa classifier loss", loss_sa, global_step=self.total_it
            )

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(
                self.target_q_funcs.parameters(), self.q_funcs.parameters()
            ):
                target_q_param.data.copy_(
                    self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data
                )

    def update_q_functions(
        self,
        state_batch,
        action_batch,
        reward_batch,
        nextstate_batch,
        not_done_batch,
        writer=None,
    ):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(
                nextstate_batch, get_logprob=True
            )
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + not_done_batch * self.discount * (
                q_target - self.alpha * logprobs_batch
            )
            # value_target = reward_batch + not_done_batch * self.discount * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar("train/q1", q_1.mean(), self.total_it)
            writer.add_scalar("train/logprob", logprobs_batch.mean(), self.total_it)
        loss = F.mse_loss(q_1, value_target) + F.mse_loss(q_2, value_target)
        return loss

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def update_tuning_ensemble(self, src_replay_buffer, batch_size, writer=None):
        """Train each network in the tuning ensemble as a dynamics model on source data.

        Each network learns to predict next_state given (state, action) from the
        source replay buffer.  Every member is trained on an independently sampled
        batch so that the ensemble maintains diversity.
        """
        ensemble_loss_total = 0.0
        for network, optimizer in zip(
            self.tuning_ensemble, self.tuning_ensemble_optimizer
        ):
            state, action, next_state, _, _ = src_replay_buffer.sample(batch_size)
            model_input = torch.cat([state, action], dim=1)
            predicted_next_state = network(model_input)
            loss = F.mse_loss(predicted_next_state, next_state)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ensemble_loss_total += loss.item()

        # Log mean ensemble loss
        if writer is not None and self.total_it % 5000 == 0:
            writer.add_scalar(
                "train/tuning_ensemble_loss",
                ensemble_loss_total / len(self.tuning_ensemble),
                global_step=self.total_it,
            )

    def train(self, src_replay_buffer, tar_replay_buffer, batch_size=128, writer=None):

        self.total_it += 1

        if (
            src_replay_buffer.size < 2 * batch_size
            or tar_replay_buffer.size < batch_size
        ):
            return

        # follow the original paper, DARC has a warmup phase that does not involve reward modification
        if self.total_it <= int(1e5):
            src_state, src_action, src_next_state, src_reward, src_not_done = (
                src_replay_buffer.sample(2 * batch_size)
            )
        else:
            if self.total_it % self.config["tar_env_interact_freq"] == 0:
                self.update_classifier(
                    src_replay_buffer, tar_replay_buffer, batch_size, writer
                )

            src_state, src_action, src_next_state, src_reward, src_not_done = (
                src_replay_buffer.sample(2 * batch_size)
            )

            # we do reward modification
            with torch.no_grad():
                sas_probs, sa_probs = self.classifier(
                    src_state, src_action, src_next_state, with_noise=False
                )
                sas_log_probs, sa_log_probs = (
                    torch.log(sas_probs + 1e-10),
                    torch.log(sa_probs + 1e-10),
                )
                reward_penalty = (
                    sas_log_probs[:, 1:]
                    - sa_log_probs[:, 1:]
                    - sas_log_probs[:, :1]
                    + sa_log_probs[:, :1]
                )

                if writer is not None and self.total_it % 5000 == 0:
                    writer.add_scalar(
                        "train/reward penalty",
                        reward_penalty.mean(),
                        global_step=self.total_it,
                    )

                src_reward += self.config["penalty_coefficient"] * reward_penalty

        q_loss_step = self.update_q_functions(
            src_state, src_action, src_reward, src_next_state, src_not_done, writer
        )

        self.q_optimizer.zero_grad()
        q_loss_step.backward()
        self.q_optimizer.step()

        self.update_target()

        # update policy and temperature parameter
        for p in self.q_funcs.parameters():
            p.requires_grad = False
        pi_loss_step, a_loss_step = self.update_policy_and_temp(src_state)
        self.policy_optimizer.zero_grad()
        pi_loss_step.backward()
        self.policy_optimizer.step()

        if self.config["temperature_opt"]:
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()

        for p in self.q_funcs.parameters():
            p.requires_grad = True

        # update tuning ensemble (source dynamics models)
        self.update_tuning_ensemble(src_replay_buffer, batch_size, writer)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, filename):
        torch.save(self.q_funcs.state_dict(), filename + "_critic")
        torch.save(self.q_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.policy.state_dict(), filename + "_actor")
        torch.save(self.policy_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.classifier.state_dict(), filename + "_classifier")
        torch.save(
            self.classifier_optimizer.state_dict(), filename + "_classifier_optimizer"
        )

    def load(self, filename):
        self.q_funcs.load_state_dict(torch.load(filename + "_critic"))
        self.q_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.policy.load_state_dict(torch.load(filename + "_actor"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.classifier.load_state_dict(torch.load(filename + "_classifier"))
        self.classifier_optimizer.load_state_dict(
            torch.load(filename + "_classifier_optimizer")
        )
