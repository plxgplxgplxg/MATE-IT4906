# Giải thích `agent.py` của MAPPO

File gốc: [gym_agent/algos/on_policy/mappo_core/agent.py](/Volumes/plxg2/Project/MATE-IT4906/gym_agent/algos/on_policy/mappo_core/agent.py)

## 1. File này dùng để làm gì

`agent.py` là trung tâm của thuật toán MAPPO trong project này. Nó phụ trách 4 việc chính:

- khởi tạo toàn bộ thành phần cần thiết của agent
- thu thập dữ liệu chạy thử từ môi trường
- cập nhật actor và critic theo mục tiêu học của PPO
- lưu, tải và suy luận action

Có thể hiểu ngắn gọn class `MAPPO` là bộ điều phối toàn bộ vòng lặp huấn luyện:

1. reset môi trường
2. actor sinh action cho từng agent
3. critic ước lượng giá trị
4. lưu transition vào buffer
5. tính return và advantage
6. tối ưu actor/critic bằng PPO

## 2. Các import quan trọng

```python
from .buffer import MAPPORolloutBuffer
from .config import MAPPOConfig
from .env import MultiCameraEnvBatch
from .networks import RecurrentGaussianActor, RecurrentValueCritic
from .normalization import ValueNormalizer
from .state import AgentSpecificStateBuilder
from .utils import masked_mean
```

Ý nghĩa:

- `MAPPORolloutBuffer`: nơi lưu dữ liệu chạy thử để huấn luyện
- `MAPPOConfig`: tập siêu tham số
- `MultiCameraEnvBatch`: tạo nhiều môi trường MATE để train song song
- `RecurrentGaussianActor`: mạng chính sách có GRU, đầu ra là hành động liên tục
- `RecurrentValueCritic`: mạng ước lượng giá trị có GRU
- `ValueNormalizer`: chuẩn hóa giá trị return và value để huấn luyện ổn định hơn
- `AgentSpecificStateBuilder`: tạo trạng thái toàn cục cho critic từ observation của nhiều agent
- `masked_mean`: tính trung bình nhưng bỏ qua phần padding trong recurrent batch

## 3. Khởi tạo class `MAPPO`

### 3.1. Constructor

```python
class MAPPO:
    def __init__(...)
```

Hàm này thực hiện toàn bộ bước setup.

### 3.2. Nạp config và seed

```python
self.config = config or MAPPOConfig()
self._validate_config()
self._seed()
```

Ý nghĩa:

- nếu người dùng không truyền config thì dùng cấu hình mặc định
- kiểm tra siêu tham số có hợp lệ không
- set seed để kết quả có thể lặp lại hơn

### 3.3. Chọn thiết bị chạy

```python
self.device = get_device(self.config.device)
```

Nếu có GPU thì có thể chạy trên GPU, nếu không thì chạy CPU.

### 3.4. Tạo môi trường và đối thủ

```python
self.state_builder = AgentSpecificStateBuilder()
self.opponent_agent_factory = opponent_agent_factory or GreedyTargetAgent
self.env_factory = env_factory or self._default_env_factory
self.env_batch = MultiCameraEnvBatch(...)
```

Ý nghĩa:

- `state_builder` tạo đầu vào cho critic
- đối thủ mặc định là `GreedyTargetAgent`
- `env_batch` chạy nhiều môi trường song song để tăng tốc thu thập dữ liệu

### 3.5. Suy ra kích thước bài toán

```python
self.num_envs = self.env_batch.num_envs
self.num_agents = self.env_batch.num_agents
self.obs_dim = self.env_batch.obs_dim
self.action_dim = self.env_batch.action_dim
self.state_dim = self.num_agents * self.obs_dim + self.obs_dim
```

Ý nghĩa:

- `num_envs`: số lượng môi trường chạy song song
- `num_agents`: số camera agent trong mỗi môi trường
- `obs_dim`: số chiều observation của mỗi agent
- `action_dim`: số chiều hành động liên tục của mỗi camera
- `state_dim`: số chiều trạng thái toàn cục cho critic

Điểm quan trọng:

- actor nhận `obs` của từng agent
- critic nhận trạng thái toàn cục

Đây là ý tưởng cốt lõi của MAPPO: actor phân tán, critic tập trung.

### 3.6. Tạo actor và critic

```python
self.actor = RecurrentGaussianActor(...)
self.critic = RecurrentValueCritic(...)
self.value_normalizer = ValueNormalizer().to(self.device)
```

Ý nghĩa:

- actor sinh phân phối hành động kiểu Gaussian
- critic dự đoán giá trị trạng thái
- cả hai đều có recurrent state để học theo chuỗi thời gian
- `value_normalizer` giúp giá trị mục tiêu ổn định hơn

### 3.7. Tạo bộ tối ưu

```python
self.actor_optimizer = torch.optim.Adam(...)
self.critic_optimizer = torch.optim.Adam(...)
```

Dùng Adam để cập nhật actor và critic.

### 3.8. Tạo bộ nhớ tạm

```python
self.buffer = MAPPORolloutBuffer(...)
```

Buffer lưu:

- observation
- trạng thái toàn cục
- action
- reward
- done
- hidden state
- giá trị dự đoán
- log probability

Đây là toàn bộ dữ liệu cần để tính GAE và hàm mất mát của PPO.

### 3.9. Trạng thái ban đầu của quá trình học

```python
self.current_obs = self.env_batch.reset()
self.current_actor_hidden = np.zeros(...)
self.current_critic_hidden = np.zeros_like(...)
self.current_episode_starts = np.ones(...)
```

Ý nghĩa:

- reset môi trường để lấy observation đầu tiên
- hidden state của GRU ban đầu bằng 0
- `episode_starts = True` vì đây là bắt đầu episode

### 3.10. Biến thống kê

```python
self.total_env_steps = 0
self.total_agent_steps = 0
self.n_updates = 0
self.completed_episodes = 0
self.last_training_stats = {}
```

Dùng để ghi lại quá trình huấn luyện.

## 4. Kiểm tra config

```python
def _validate_config(self) -> None:
```

Hàm này chặn các giá trị vô lý như:

- `num_envs <= 0`
- `rollout_length <= 0`
- `recurrent_chunk_length <= 0`
- `n_epochs <= 0`
- `num_mini_batches <= 0`

Nếu không kiểm tra sớm, code train sẽ lỗi hoặc cho kết quả vô nghĩa.

## 5. Seed

```python
def _seed(self) -> None:
```

Nếu có `seed`, hàm sẽ set:

- `np.random.seed`
- `torch.manual_seed`

Mục đích là giúp kết quả huấn luyện bớt ngẫu nhiên giữa các lần chạy.

## 6. Tạo môi trường mặc định

```python
def _default_env_factory(self) -> gym.Env:
```

Hàm này:

1. gọi `gym.make(...)` để tạo MATE env
2. bọc env bằng `mate.MultiCamera.make(...)`
3. gán đối thủ `GreedyTargetAgent`

Ý nghĩa:

- agent được huấn luyện là team camera
- team target được điều khiển bởi luật heuristic

## 7. Hàm `_torch`

```python
def _torch(self, array, dtype=None) -> torch.Tensor:
```

Đây là helper để:

- chuyển `numpy` sang `torch.Tensor`
- đưa dữ liệu lên `device`
- ép `dtype` nếu cần

Hàm này giúp code trong phần chạy thử và cập nhật gọn hơn.

## 8. Hai hàm liên quan đến value normalization

### 8.1. `_denormalize_values`

```python
def _denormalize_values(self, raw_values):
```

Nếu critic cho ra giá trị đã được chuẩn hóa, hàm này đưa nó về thang đo thật.

### 8.2. `_value_targets`

```python
def _value_targets(self, returns):
```

Nếu bật `use_value_normalization`, return sẽ được chuẩn hóa trước khi tính sai số của critic.

Ý tưởng ở đây là:

- critic học trên mục tiêu đã chuẩn hóa
- nhưng khi tính GAE/return thì vẫn cần giá trị theo scale thật

## 9. `close`

```python
def close(self) -> None:
    self.env_batch.close()
```

Đóng toàn bộ môi trường.

## 10. `collect_rollout`: thu thập dữ liệu chạy thử

Đây là phần quan trọng nhất của giai đoạn tương tác với môi trường.

```python
def collect_rollout(self) -> None:
    self.buffer.reset()
```

Bắt đầu đợt chạy thử mới thì xóa bộ nhớ tạm cũ.

### 10.1. Lặp qua `rollout_length`

```python
for _ in range(self.config.rollout_length):
```

Mỗi vòng lặp tương ứng 1 timestep trên tất cả môi trường.

### 10.2. Tạo trạng thái toàn cục cho critic

```python
global_states = self.state_builder(self.current_obs)
```

Actor chỉ thấy observation của chính nó. Critic được cấp thông tin rộng hơn, nên cần `global_states`.

Trong MAPPO:

- actor: phân tán
- critic: tập trung

### 10.3. Flatten dữ liệu

```python
flat_obs = self.current_obs.reshape(...)
flat_states = global_states.reshape(...)
flat_actor_hidden = self.current_actor_hidden.reshape(...)
flat_critic_hidden = self.current_critic_hidden.reshape(...)
flat_episode_starts = self.current_episode_starts.reshape(...)
```

Lý do:

- dữ liệu gốc có dạng `(num_envs, num_agents, ...)`
- network muốn xử lý theo batch 2 chiều nên cần flatten thành `(num_envs * num_agents, ...)`

### 10.4. Actor chọn hành động, critic dự đoán giá trị

```python
with torch.no_grad():
    action_tensor, log_prob_tensor, next_actor_hidden = self.actor.act(...)
    raw_value_tensor, next_critic_hidden = self.critic.predict_values(...)
    value_estimate_tensor = self._denormalize_values(raw_value_tensor)
```

Ý nghĩa:

- `actor.act`: sinh hành động theo chính sách hiện tại
- `log_prob_tensor`: lưu lại để sau này tính tỉ lệ PPO
- `critic.predict_values`: dự đoán giá trị trạng thái
- `torch.no_grad()`: vì đang chạy thử, chưa học

Phần này là giai đoạn thu thập trải nghiệm, không phải giai đoạn cập nhật trọng số.

### 10.5. Đưa tensor về numpy

```python
actions = action_tensor.cpu().numpy().reshape(...)
log_probs = log_prob_tensor.cpu().numpy().reshape(...)
value_preds = raw_value_tensor.cpu().numpy().reshape(...)
value_estimates = value_estimate_tensor.cpu().numpy().reshape(...)
```

Môi trường cần `numpy`, không cần `torch.Tensor`.

### 10.6. Step môi trường

```python
next_obs, rewards, dones, next_episode_starts, _ = self.env_batch.step(actions)
```

Sau khi có action:

- môi trường trả về observation mới
- reward cho mỗi agent
- done
- có episode mới hay không

### 10.7. Reset hidden state khi episode kết thúc

```python
env_done_mask = dones[:, :1].astype(np.float32)
next_actor_hidden_np *= 1.0 - env_done_mask[..., None]
next_critic_hidden_np *= 1.0 - env_done_mask[..., None]
```

Ý nghĩa:

- nếu môi trường kết thúc, hidden state cũ không được mang sang episode mới
- cần đưa nó về 0 để tránh "nhớ nhầm" giữa 2 episode

### 10.8. Lưu transition vào buffer

```python
self.buffer.add(...)
```

Bộ nhớ tạm sẽ lưu toàn bộ dữ liệu cần cho bước cập nhật PPO.

### 10.9. Cập nhật biến hiện tại

```python
self.current_obs = next_obs
self.current_actor_hidden = next_actor_hidden_np
self.current_critic_hidden = next_critic_hidden_np
self.current_episode_starts = next_episode_starts
```

Để bước timestep tiếp theo dùng đúng trạng thái mới.

### 10.10. Bootstrap value ở timestep cuối

Sau khi chạy thử xong, code cần giá trị của trạng thái cuối cùng để tính GAE:

```python
final_values, _ = self.critic.predict_values(...)
final_values = self._denormalize_values(final_values)
```

Nếu không có giá trị bootstrap, ta không tính đúng return và advantage cho bước cuối.

### 10.11. Tính GAE và return

```python
self.buffer.compute_returns_and_advantages(...)
if self.config.normalize_advantage:
    self.buffer.normalize_advantages()
```

Ý nghĩa:

- tính advantage bằng GAE
- tính return = advantage + giá trị dự đoán
- có thể normalize advantage để train ổn định hơn

### 10.12. Đếm số step

```python
self.total_env_steps += ...
self.total_agent_steps += ...
```

Khác nhau:

- `env_steps`: đếm số bước của môi trường
- `agent_steps`: đếm tổng bước của tất cả agent

## 11. `update`: học từ dữ liệu đã thu thập

```python
def update(self) -> dict[str, float]:
```

Đây là giai đoạn tối ưu trọng số.

### 11.1. Cập nhật bộ chuẩn hóa giá trị

```python
if self.config.use_value_normalization:
    self.value_normalizer.update(...)
```

Bộ chuẩn hóa giá trị cần nhìn thấy return mới để cập nhật trung bình và độ lệch chuẩn đang chạy.

### 11.2. Vòng lặp epoch và minibatch

```python
for _ in range(self.config.n_epochs):
    for batch in self.buffer.iterate_recurrent_batches(...):
```

Ý nghĩa:

- 1 đợt dữ liệu có thể học lại nhiều lần
- recurrent batch được cắt theo chunk để train GRU

### 11.3. Chuyển batch sang tensor

```python
observations = self._torch(batch.observations, dtype=torch.float32)
...
loss_mask = self._torch(batch.loss_mask, dtype=torch.float32)
```

`loss_mask` rất quan trọng:

- chunk recurrent có thể bị padding
- mask đảm bảo chỉ tính loss trên timestep hợp lệ

### 11.4. Tính xác suất mới và entropy mới

```python
new_log_probs, entropy = self.actor.evaluate_actions(...)
new_values, _ = self.critic.forward(...)
```

Ý nghĩa:

- chạy lại actor trên cùng observation và action cũ
- lấy `new_log_probs` để so với `old_log_probs`
- critic tính giá trị mới

Do đây là PPO, ta cần biết chính sách mới đã thay đổi bao nhiêu so với chính sách cũ.

### 11.5. Mục tiêu PPO có chặn

```python
ratio = torch.exp(new_log_probs - old_log_probs)
surrogate_1 = ratio * advantages
surrogate_2 = ratio.clamp(
    1.0 - self.config.clip_range,
    1.0 + self.config.clip_range,
) * advantages
actor_loss = -masked_mean(
    torch.min(surrogate_1, surrogate_2),
    loss_mask,
) - self.config.entropy_coef * masked_mean(entropy, loss_mask)
```

Đây là công thức cốt lõi của PPO.

Ý nghĩa từng phần:

- `ratio`: mức độ chính sách mới khác chính sách cũ
- `surrogate_1`: mục tiêu thông thường
- `surrogate_2`: mục tiêu đã bị chặn
- `min(...)`: ngăn cập nhật chính sách quá mạnh
- trừ entropy: khuyến khích khám phá

Trong thuyết trình, có thể nói ngắn gọn:

"Actor được học theo PPO có chặn để cải thiện chính sách nhưng không thay đổi quá đột ngột."

### 11.6. Sai số giá trị có chặn

```python
value_targets = self._value_targets(returns)
clipped_values = old_value_preds + (
    new_values - old_value_preds
).clamp(-self.config.clip_range, self.config.clip_range)
unclipped_value_loss = F.huber_loss(...)
clipped_value_loss = F.huber_loss(...)
critic_loss = masked_mean(
    torch.max(unclipped_value_loss, clipped_value_loss),
    loss_mask,
)
```

Ý nghĩa:

- critic học dự đoán `returns`
- giá trị cũng được chặn để tránh cập nhật quá mạnh
- dùng `Huber loss` thay vì MSE để bớt nhạy với outlier

Đây là kiểu chặn giá trị phổ biến trong nhiều bản PPO/MAPPO.

### 11.7. Tổng sai số và lan truyền ngược

```python
total_loss = actor_loss + self.config.value_coef * critic_loss

self.actor_optimizer.zero_grad(set_to_none=True)
self.critic_optimizer.zero_grad(set_to_none=True)
total_loss.backward()
nn.utils.clip_grad_norm_(...)
self.actor_optimizer.step()
self.critic_optimizer.step()
```

Ý nghĩa:

- tổng sai số = sai số actor + hệ số * sai số critic
- xóa gradient cũ
- lan truyền ngược
- clip gradient để tránh nổ gradient
- cập nhật tham số

### 11.8. Lưu thống kê huấn luyện

Code cộng dồn:

- `actor_loss_total`
- `critic_loss_total`
- `entropy_total`

Sau đó chia cho `batch_count` và trả về.

## 12. `learn`: vòng lặp huấn luyện tổng quát

```python
def learn(self, total_env_steps: int) -> dict[str, float]:
    while self.total_env_steps < total_env_steps:
        self.collect_rollout()
        self.update()
    return self.last_training_stats
```

Rất đơn giản:

1. thu thập dữ liệu
2. cập nhật mô hình
3. lặp lại đến khi đủ số step

Đây là vòng lặp huấn luyện mức cao nhất.

## 13. `predict`: suy luận hành động sau khi huấn luyện

```python
def predict(...)
```

Hàm này dùng khi:

- kiểm tra mô hình
- chạy thử suy luận
- demo

### 13.1. Kiểm tra shape observation

```python
if observation.shape != (self.num_agents, self.obs_dim):
    raise ValueError(...)
```

Đảm bảo input đúng định dạng.

### 13.2. Xử lý hidden state ban đầu

Nếu người dùng không đưa `actor_hidden_state`, code tự tạo hidden bằng 0.

Nếu không đưa `episode_start`, code giả sử đây là đầu episode.

### 13.3. Gọi actor

```python
with torch.no_grad():
    actions, _, next_hidden = self.actor.act(...)
```

Trả về:

- `actions`: hành động cho mỗi agent
- `next_hidden`: hidden state mới để đưa sang timestep sau

Nếu `deterministic=True`, hành động sẽ theo giá trị trung bình thay vì lấy mẫu.

## 14. `save`: lưu mốc huấn luyện

```python
def save(self, path: str | Path) -> None:
```

Code lưu:

- config
- trọng số actor
- trọng số critic
- trạng thái bộ tối ưu
- bộ chuẩn hóa giá trị
- trạng thái huấn luyện

Việc này cho phép tiếp tục huấn luyện hoặc kiểm tra sau này.

## 15. `load`: tải mốc huấn luyện

```python
@classmethod
def load(...)
```

Quy trình:

1. `torch.load(...)`
2. tạo lại agent từ config đã lưu
3. nạp trọng số actor và critic
4. nạp bộ tối ưu
5. phục hồi thống kê huấn luyện

Sau đó có thể:

- huấn luyện tiếp
- evaluate
- suy luận hành động

## 16. Bản chất thuật toán trong file này

Nếu bỏ qua chi tiết code, file này đang triển khai ý tưởng sau:

1. Mỗi agent nhìn `observation` riêng của nó
2. Actor recurrent sinh hành động liên tục cho mỗi agent
3. Critic recurrent nhìn `global state` rộng hơn để đánh giá giá trị
4. Thu thập dữ liệu chạy thử trong nhiều môi trường song song
5. Tính GAE và returns
6. Cập nhật bằng PPO có chặn
7. Lặp lại

Đó là MAPPO theo đúng tinh thần:

- đa tác tử
- dùng chung chính sách
- critic tập trung
- thực thi phân tán
- cập nhật kiểu PPO

## 17. Các ý có thể dùng khi thuyết trình

Bạn có thể trình bày theo 5 ý chính:

### 17.1. Vì sao dùng MAPPO

- có nhiều agent camera cùng hoạt động
- mỗi agent có observation riêng
- cần một critic trung tâm để học ổn định hơn

### 17.2. Vì sao actor và critic đều recurrent

- bài toán diễn ra theo chuỗi thời gian
- observation tại một thời điểm có thể chưa đủ thông tin
- hidden state giúp nhớ lịch sử gần đây

### 17.3. Vì sao critic dùng global state

- critic cần bối cảnh đầy đủ hơn để ước lượng giá trị tốt hơn
- actor vẫn giữ thực thi phân tán

### 17.4. Vì sao dùng PPO clip

- tránh cập nhật chính sách quá mạnh
- huấn luyện ổn định hơn

### 17.5. Vì sao cần buffer và GAE

- buffer lưu dữ liệu chạy thử
- GAE giúp cân bằng bias và variance khi tính advantage

## 18. Một số lưu ý kỹ thuật

- File này đang tối ưu cho hành động liên tục, vì actor dùng chính sách Gaussian.
- Chính sách dường như được chia sẻ giữa các agent, vì tất cả agent đi qua cùng một actor network sau khi flatten batch.
- `dones[:, :1]` được dùng để reset hidden theo môi trường, giả định rằng các agent trong cùng môi trường kết thúc cùng lúc.
- `state_dim = num_agents * obs_dim + obs_dim` cho thấy `state_builder` đang tạo một trạng thái toàn cục có tính phụ thuộc vào từng agent, không phải một trạng thái chung duy nhất rất đơn giản.

## 19. Cách hiểu nhanh toàn bộ file trong 1 câu

`agent.py` là file điều phối việc dùng actor-critic recurrent để thu thập dữ liệu trong bài toán đa tác tử, sau đó cập nhật chính sách bằng MAPPO/PPO với critic tập trung và GAE.
