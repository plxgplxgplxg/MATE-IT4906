# Q-MARL: Tổng Hợp Toàn Diện

> **Q-MARL — A Quantum-Inspired Algorithm Using Neural Message Passing for Large-Scale Multi-Agent Reinforcement Learning**
> Tác giả: Kha Vo & Chin-Teng Lin | arXiv:2503.07397v1 | 10/03/2025

---

## 1. Tổng Quan

Q-MARL là kiến trúc học phi tập trung hoàn toàn (fully decentralised), lấy cảm hứng từ kỹ thuật dự đoán thuộc tính phân tử trong **hóa học lượng tử** (quantum chemistry). Mục tiêu: giải quyết bài toán MARL quy mô rất lớn (hàng nghìn agent) mà không cần các giả định mạnh như phần thưởng chung hay thứ tự agent.

### Điểm khác biệt cốt lõi
| Đặc điểm | Phương pháp cũ | Q-MARL |
|---|---|---|
| Số agent tối đa thực tế | ~50 | Hàng nghìn (đã test 10.000) |
| Kiểu học | Tập trung / bán tập trung | Phi tập trung hoàn toàn |
| Xử lý không gian trạng thái | Toàn cục (bùng nổ tổ hợp) | Sub-graph cục bộ |
| Đảm bảo hội tụ | Không | Có (chứng minh lý thuyết) |
| Graph isomorphism | Không xét | Invariant (xoay/đối xứng đồ thị) |

---

## 2. Nền Tảng Lý Thuyết

### 2.1 Ký Hiệu & Không Gian

| Ký hiệu | Ý nghĩa |
|---|---|
| $N$ | Tổng số agent trong môi trường |
| $S^i$ | Không gian trạng thái hữu hạn của agent $i$ |
| $A^i$ | Không gian hành động hữu hạn của agent $i$ |
| $S = \prod_i S^i$ | Không gian trạng thái chung (joint state space) |
| $A = \prod_i A^i$ | Không gian hành động chung (joint action space) |
| $S^i_t$ | Trạng thái tức thời của agent $i$ tại bước $t$ |
| $A^i_t$ | Hành động của agent $i$ tại bước $t$ |
| $R^i_{t+1}$ | Phần thưởng agent $i$ nhận được sau bước $t$ |
| $\theta^i$ | Tham số chính sách của agent $i$ |
| $\gamma \in (0, 1]$ | Hệ số giảm (discount factor) |
| $G_t = \sum_{t=0}^{\infty} \gamma^t R_{t+1}$ | Phần thưởng tích lũy dài hạn |
| $G(V, E)$ | Đồ thị đầy đủ: $V$ = tập đỉnh (agent), $E$ = tập cạnh |
| $G^i$ | Sub-graph từ góc nhìn agent $i$ |
| $\mathcal{N}(i)$ | Tập hàng xóm của agent $i$ |
| $d_{ij}$ | Khoảng cách giữa agent $i$ và $j$ |
| $z_{ij}$ | Vector đặc trưng cạnh (edge feature) giữa $i$ và $j$ |
| $s^i$ | Vector đặc trưng trạng thái của agent $i$ |
| $s^l_i$ | Hidden feature của agent $i$ tại tầng $l$ |
| $w^i$ | Tham số mạng critic của agent $i$ |
| $\hat{q}^i(s,a)$ | Hàm action-value ước lượng cục bộ của agent $i$ |

---

### 2.2 Công Thức Chính Sách (Policy)

**Chính sách tham số hóa:**

$$\pi^i(a^i | s) = P\left(A^i_t = a^i \mid S_t = s,\ \theta^i\right) \tag{2.1}$$

**Mục tiêu tối ưu hóa:**

$$\tilde{\theta} = \underset{\theta}{\arg\max}\ \lambda(\theta) \triangleq \mathbb{E}[G_t \mid \theta] \tag{2.2}$$

Với phần thưởng trung bình toàn cục: $R_t = \frac{1}{N}\sum_i R^i_t$

---

### 2.3 Định Lý 1 — Policy Gradient trong MARL

**Phát biểu:** Gradient của phần thưởng toàn cục $\lambda(\theta)$ theo tham số cục bộ $\theta^i$:

$$\nabla_{\theta^i} \lambda(\theta) = \mathbb{E}_\pi\left[G_t \nabla_{\theta^i} \ln \pi^i(A^i_t \mid S_t)\right] \tag{2.3}$$

**Chứng minh (tóm tắt):**
Từ Policy Gradient theorem gốc [Sutton et al., 1999]:

$$\nabla \lambda(\theta) = \mathbb{E}_\pi\left[G_t \frac{\nabla \pi(A_t|S_t)}{\pi(A_t|S_t)}\right] \tag{2.4}$$

Vì $\pi(A_t|S_t) = \prod_i \pi^i(A^i_t|S_t)$ và các chính sách độc lập có điều kiện:
$$\nabla_{\theta^i} \ln \pi^j(A^j_t|S_t) = 0 \quad \forall j \neq i$$

---

### 2.4 Định Lý 2 — Cập Nhật Chính Sách Cục Bộ

**Quy tắc cập nhật:**

$$\theta^i \leftarrow \theta^i + \alpha \delta^i \nabla_{\theta^i} \ln \pi^i(A^i_t|S_t) \tag{2.5}$$

Trong đó:
$$\delta^i = \hat{q}^i(S_t, A_t) - v(S_t, A^{(-i)}_t)$$

- $A^{(-i)}_t$: hành động của tất cả agent **độc lập** với agent $i$ tại bước $t$
- $v(S_t, A^{(-i)}_t)$: baseline offset để giảm phương sai

**Baseline được tính bằng:**

$$v^i(S_t, A^{(-i)}_t) = \sum_{a^i} \pi^i(a^i|S_t) \cdot \hat{q}^i\left(S_t, \left(A^{(1)}_t, \ldots, a^i, \ldots, A^{(N)}_t\right)\right)$$

---

### 2.5 Định Lý 3 — Ensemble Action Cải Tiến

**Phát biểu:** Hành động ensemble của agent $i$ (trung bình qua tất cả sub-graph chứa nó) tốt hơn bất kỳ hành động đơn lẻ nào:

$$\pi_{\text{ensemble}}(x, P) = \mathbb{E}_G[\pi(x, G)] \tag{2.7}$$

**Chứng minh:**
- Sai số trung bình của hành động đơn: $e = \mathbb{E}_G \mathbb{E}_{X,Y}[Y - \pi(X,G)]^2$
- Sai số ensemble: $e_{\text{ensemble}} = \mathbb{E}_{X,Y}[Y - \pi_{\text{ensemble}}(X,P)]^2$
- Vì $\mathbb{E}[Z^2] \geq (\mathbb{E}[Z])^2$, suy ra:

$$e \geq e_{\text{ensemble}} \tag{2.10}$$

$$e - e_{\text{ensemble}} \geq \mathbb{E}_{X,Y}\left[\text{Var}_G\ \pi(X,G)\right] \tag{2.11}$$

---

## 3. Kiến Trúc Neural Message Passing (NMP)

### 3.1 Sơ Đồ Tổng Quan

```
Input: s_i (state features), z_ij (edge features / distances)
  │
  ▼
[rbe] Radial Basis Expansion → z_ij encoded (10-dim)
  │
  ▼
Lặp qua L tầng:
  ┌──────────────────────────────────────────┐
  │  Vertex Update Block (V)                  │
  │    s^{l+1}_i = V(s^l_i, z^l_ij)          │
  │    = hlin( hrel(s^l_i) + Σ_j hrel(z^l_ij)) │
  │                                            │
  │  Edge Update Block (E)                    │
  │    z^{l+1}_ij = E(s^{l+1}_i, s^{l+1}_j, z^l_ij) │
  │    = hrel(concat(s^{l+1}_i, s^{l+1}_j, z^l_ij)) │
  └──────────────────────────────────────────┘
  │
  ▼
Output: π^i_j (action distribution) cho mỗi agent j trong sub-graph G^i
```

### 3.2 Các Khối Xây Dựng

| Ký hiệu | Tên | Công thức |
|---|---|---|
| `hlin` | Linear layer | $h_{\text{lin}}(x) = Wx + b$ |
| `hrel` | ReLU layer | $h_{\text{rel}}(x) = \max(0, Wx + b)$ |
| `rbe` | Radial Basis Expansion | $z_{ij,n} = \exp\left(-\frac{(d_{ij} - n\Delta d)^2}{\Delta d}\right)$ |
| `concat` | Concatenation | Nối vector |
| `Σ_j` | Summation | Tổng các cạnh trong $\mathcal{N}(i)$ |
| `ebd` | Edge embedding | FC layer trên edge features |

### 3.3 Radial Basis Expansion (RBE) — Chi Tiết

$$z_{ij,n} = \exp\left(-\frac{(d_{ij} - n\Delta d)^2}{\Delta d}\right), \quad n = 0, 1, \ldots, n_{\max} \tag{2.14}$$

| Tham số RBE | Giá trị | Mô tả |
|---|---|---|
| $n_{\max}$ | **10** | Số chiều của vector mã hóa khoảng cách |
| $n\Delta d$ | = max view range | Bước cuối bằng tầm quan sát tối đa |
| $\Delta d$ | = max\_range / $n_{\max}$ | Bước nhảy |

### 3.4 Vertex & Edge Update

**Vertex Update (V):**
$$s^{l+1}_i = h_{\text{lin}}\left(h_{\text{rel}}(s^l_i) + \sum_{j \in \mathcal{N}(i)} h_{\text{rel}}(z^l_{ij})\right) + s^{l+1}_i \quad \text{(residual)} \tag{2.12}$$

**Edge Update (E):**
$$z^{l+1}_{ij} = h_{\text{rel}}\left(\text{concat}(s^{l+1}_i,\ s^{l+1}_j,\ z^l_{ij})\right) \tag{2.13}$$

---

## 4. Thuật Toán Chính (Algorithm 1)

```
GRAPH-BASED MARL (Q-MARL)

Khởi tạo:
  - π(a|s)  : mô hình chính sách graph, tham số θ
  - q̂(s,a)  : hàm action-value (critic), tham số w
  - α_θ, α_w : learning rates

FOR mỗi episode:
  FOR mỗi time step:
    IF điều kiện kết thúc episode → BREAK

    1. [DECOMPOSE]
       G^i ← decompose(S)
       // Phân rã trạng thái toàn cục S thành N sub-graph
       // mỗi G^i là sub-graph từ góc nhìn agent i

    2. [SAMPLE ACTIONS]
       Lấy mẫu A^i_j ~ π(G^i) với mọi j liên quan trong G^i

    3. [ENSEMBLE]
       A^i = Σ_j A^i_j  (trung bình/tổng hành động) với mỗi agent i

    4. [EXECUTE]
       Thực hiện hành động A = [A^1, ..., A^N] lên môi trường

    5. [OBSERVE]
       Quan sát G' = [G'^1, ..., G'^N],  R = [R^1, ..., R^N]

    6. [UPDATE]
       FOR mỗi i:
         FOR mỗi j liên quan trong G^i:

           // Tính baseline (value estimate hiện tại)
           v  ← Σ_j π(a_j|G^i) · q̂(G^i, [A^i_1,...,a^i_j,...,A^i_end])

           // Tính baseline (value estimate bước kế)
           v' ← Σ_j π(a_j|G'^i) · q̂(G'^i, [A^i_1,...,a^i_j,...,A^i_end])

           // TD error (advantage)
           δ ← R^i + γv' - v

           // Cập nhật critic
           w ← w + α_w · δ · ∇_w q̂(G^i, A^i)

           // Cập nhật actor
           θ ← θ + α_θ · δ · ∇_θ ln π(A^i_j | G^i)
```

---

## 5. Cấu Hình Tốt Nhất (Best Configurations)

### 5.1 Siêu Tham Số Mô Hình

| Tham số | Giá trị tốt nhất | Mô tả |
|---|---|---|
| `depth` (bậc sub-graph) | **3** | Bậc láng giềng (3rd-degree neighbours); kiểm soát độ phức tạp |
| `nmax` (RBE dimensions) | **10** | Số chiều mã hóa khoảng cách |
| `n_delta_d` | = max view range | Bước tăng trong RBE |
| Observation window | **3×3** | Vùng quan sát cục bộ quanh mỗi agent |
| Action space size | **5** | {lên, xuống, trái, phải, đứng yên} |

### 5.2 Cấu Hình Huấn Luyện

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Optimizer | **Adam** | [Kingma & Ba, 2014] |
| Learning rate ban đầu | **0.01** | Cho cả actor và critic |
| LR scheduler | **Reduce-on-Plateau** | Giảm khi không cải thiện |
| LR decay rate | **5%** | Giảm 5% sau mỗi lần trigger |
| LR decay patience | **10 batches** | Không cải thiện reward sau 10 batch |
| Batch size | **100 games** | 1 batch = 100 game/episode |
| Discount factor $\gamma$ | $(0, 1]$ | Mặc định gần 1 cho long-horizon |

### 5.3 Biến Thể Thuật Toán

| Biến thể | Mô tả | Hiệu suất |
|---|---|---|
| **Q-MARL-AC** | Actor-Critic + Graph NMP | **Tốt nhất** (Jungle: 91.16%, Battle: 90.16%, Deception: 35.90%) |
| **Q-MARL-PG** | Policy Gradient + Graph NMP | Tốt (Jungle: 64.86%, Battle: 87.13%, Deception: 28.91%) |
| Vanilla-PG (baseline) | Policy Gradient thuần | Kém (Jungle: 22.66%, Battle: 30.00%, Deception: −21.87%) |

**→ Khuyến nghị: Dùng Q-MARL-AC cho hầu hết các tình huống.**

---

## 6. Biến Môi Trường (Environment Variables)

### 6.1 Cấu Trúc Môi Trường Chung

| Biến | Kiểu | Mô tả |
|---|---|---|
| `N` | int | Số lượng agent |
| `grid_size` | tuple | Kích thước lưới (grid world) |
| `obs_range` | int | Tầm quan sát = 3×3 quanh agent |
| `max_steps` | int | Số bước tối đa mỗi episode |
| `allow_overlap` | bool | Cho phép nhiều agent cùng ô |
| `agent_type` | str | Homogeneous (dùng chung policy) |

### 6.2 Kết Nối Đồ Thị

| Biến | Điều kiện | Ý nghĩa |
|---|---|---|
| Edge $(i,j)$ tồn tại | agent $j$ nằm trong vùng 3×3 của agent $i$ | Hai agent "hàng xóm" trực tiếp |
| Sub-graph $G^i$ | Depth = k | Gồm agent $i$ + tất cả hàng xóm đến bậc k |

### 6.3 Biến Đặc Trưng Mô Hình

| Biến | Kích thước | Mô tả |
|---|---|---|
| `s_i` | $d_s$ | Vector trạng thái thô của agent $i$ |
| `z_ij` | $n_{\max} = 10$ | Vector khoảng cách mã hóa qua RBE |
| `s^l_i` | $d_h$ | Hidden feature tầng $l$ |
| `W`, `b` | $d_h \times d_{\text{in}}$ | Trọng số và bias FC layer |

---

## 7. Ba Kịch Bản Thực Nghiệm

### 7.1 Jungle (Hợp Tác)

| Thông số | Giá trị |
|---|---|
| Loại | Collaborative (social dilemma) |
| Mục tiêu | Tối đa hóa thức ăn thu thập được |
| Reward tức thời | +1 nếu đứng cạnh ô thức ăn |
| Kill condition | Bị giết nếu tiếp giáp ≥1 agent khác **3 lần** |
| Thức ăn | Cố định, tồn tại vĩnh viễn |
| Win rate Q-MARL-AC | **91.16%** |

### 7.2 Battle (Cạnh Tranh)

| Thông số | Giá trị |
|---|---|
| Loại | Competitive (2 teams, N vs N) |
| Mục tiêu | Có nhiều agent sống sót nhất sau episode |
| Kill condition | Bị vây bởi ≥3 agent đối địch |
| Reward | +1 khi thắng, −1 khi thua (cuối episode) |
| Không có reward tức thời | ✓ |
| Win rate Q-MARL-AC | **90.16%** |

### 7.3 Deception (Hỗn Hợp)

| Thông số | Giá trị |
|---|---|
| Loại | Mixed cooperative-competitive |
| Số agent home | N |
| Số adversary | 1 |
| Số landmark | Nhiều (1 là target) |
| Tấn công | Không cho phép |
| Reward home | +1 nếu adversary chưa tìm được target VÀ có ≥1 home agent ở target |
| Reward adversary | +1 nếu tìm được đúng target |
| Win rate Q-MARL-AC | **35.90%** (kịch bản khó nhất) |

---

## 8. So Sánh Với Phương Pháp Khác

### 8.1 Khả Năng Tổng Quát (Curriculum Learning)

> *Train với N nhỏ → Test với 2N agent*

| Phương pháp | Jungle N=8 | Jungle N=28 | Battle N=2 | Deception N=2 |
|---|---|---|---|---|
| RGE [27] | ~0.667 | ~0.681 | ~0.252 | ~0.210 |
| SHA [3] | ~0.746 | ~0.699 | −0.029 | −0.294 |
| MHA [19] | ~0.780 | ~0.716 | 0.107 | −0.113 |
| **Q-MARL** | **~0.911** | **~0.848** | −0.163 | −0.196 |

### 8.2 Tốc Độ Huấn Luyện (giây/100 episodes)

| Số agent | RGE | SHA | MHA | **Q-MARL (NMP)** |
|---|---|---|---|---|
| 10 | ~100 | ~100 | ~100 | ~100 |
| 100 | **~450** | ~200 | ~210 | **~200** |
| 1,000 | **>1000 (phát nổ)** | ~600 | ~620 | **~220** |
| 10,000 | **Không khả thi** | ~900 | ~860 | **~420** |

**→ Q-MARL là phương pháp DUY NHẤT có thể xử lý 10.000 agent trong thời gian hợp lý.**

---

## 9. Cài Đặt & Tái Tạo Thực Nghiệm

### 9.1 Repository

```bash
git clone https://github.com/cibciuts/NMP_MARL
cd NMP_MARL
```

### 9.2 Yêu Cầu (Suy Luận Từ Paper)

```bash
# Python dependencies (suy luận từ các component được dùng)
pip install torch          # PyTorch (neural network backbone)
pip install torch-geometric # Graph neural network support
pip install numpy
pip install matplotlib     # Visualization
```

### 9.3 Cấu Hình Optimizer (Adam)

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',          # Maximize reward
    factor=0.95,         # Giảm 5% mỗi lần trigger
    patience=10,         # Sau 10 batch không cải thiện
    verbose=True
)
```

### 9.4 RBE Encoding

```python
import numpy as np

def radial_basis_expansion(d_ij, n_max=10, view_range=3.0):
    """
    Mã hóa khoảng cách thành vector n_max chiều.
    d_ij: khoảng cách giữa agent i và j
    n_max: số chiều (mặc định 10)
    view_range: tầm quan sát tối đa
    """
    delta_d = view_range / n_max
    n_vals = np.arange(n_max)
    z = np.exp(-((d_ij - n_vals * delta_d) ** 2) / delta_d)
    return z  # shape: (n_max,)
```

### 9.5 Graph Decomposition

```python
def decompose_to_subgraphs(global_state, agents, obs_range=1):
    """
    Phân rã môi trường toàn cục thành N sub-graph.
    Mỗi sub-graph G^i được xây từ góc nhìn agent i,
    bao gồm i và tất cả hàng xóm trong obs_range bước.
    """
    subgraphs = {}
    for i, agent in enumerate(agents):
        neighbors = get_neighbors(agent, agents, obs_range)
        subgraphs[i] = build_subgraph(agent, neighbors, global_state)
    return subgraphs

def get_neighbors(agent, all_agents, depth=3):
    """BFS tìm hàng xóm đến bậc 'depth'."""
    visited = {agent.id}
    frontier = [agent]
    for _ in range(depth):
        next_frontier = []
        for a in frontier:
            for b in all_agents:
                if b.id not in visited and is_adjacent(a, b):
                    visited.add(b.id)
                    next_frontier.append(b)
        frontier = next_frontier
    return [a for a in all_agents if a.id in visited]
```

### 9.6 Action Ensemble

```python
def ensemble_actions(subgraph_actions, agent_id):
    """
    Tổng hợp hành động của agent_id từ tất cả sub-graph chứa nó.
    subgraph_actions: dict {subgraph_id: {agent_id: action_prob}}
    """
    all_actions = []
    for sg_id, actions in subgraph_actions.items():
        if agent_id in actions:
            all_actions.append(actions[agent_id])
    
    if not all_actions:
        return None
    
    # Ensemble = trung bình xác suất hành động
    ensemble = sum(all_actions) / len(all_actions)
    return ensemble
```

---

## 10. Tóm Tắt Các Đóng Góp Chính

1. **Công thức hóa MARL dưới dạng đồ thị** — Môi trường toàn cục được biểu diễn như đồ thị thay đổi theo thời gian; mỗi agent là một đỉnh, kết nối tạm thời là cạnh. Chứng minh hội tụ tới tối ưu toàn cục với giả định đồ thị time-varying.

2. **Mô hình NMP với bất biến xoay (rotational invariance)** — Lấy cảm hứng từ hóa học lượng tử; xử lý isomorphism đồ thị; tất cả đỉnh và cạnh tương tác đầy đủ trong sub-graph.

3. **Action Ensembling** — Mỗi agent xuất hiện trong nhiều sub-graph; hành động được tổng hợp (ensemble) từ tất cả, được chứng minh toán học là tốt hơn bất kỳ hành động đơn nào.

4. **Scalability** — Q-MARL xử lý được **10.000 agent** trong khi các phương pháp khác thất bại ở ~100 agent (RGE) hoặc bắt đầu chậm đáng kể ở ~1.000 (SHA, MHA).

5. **Không cần giả định mạnh** — Không cần common reward, không cần agent ordering, không cần central manager lúc test.

---

## 11. Hạn Chế Đã Biết

- **Deception scenario**: Q-MARL kém hơn các phương pháp khác trong kịch bản này. Nguyên nhân: các sub-graph riêng lẻ không thể phối hợp phân tán agent tới nhiều landmark khác nhau (yêu cầu thông tin toàn cục).
- **Không thực nghiệm với môi trường liên tục** (continuous action/state spaces).
- **Homogeneous agents only**: tất cả agent trong cùng nhóm dùng chung một policy.

---

## 12. Tài Liệu Tham Khảo Quan Trọng

| Ref | Công trình | Liên quan |
|---|---|---|
| [14] | Gilmer et al., ICML 2017 — Neural Message Passing for QC | Kiến trúc NMP gốc |
| [26] | Lowe et al., NIPS 2017 — MADDPG | Baseline MARL |
| [46] | Sutton et al., NIPS 1999 — Policy Gradient | Nền tảng lý thuyết |
| [23] | Konda & Tsitsiklis, 2000 — Actor-Critic | Thuật toán AC |
| [22] | Kingma & Ba, 2014 — Adam | Optimizer |
| [48] | Yang et al., ICML 2018 — Mean Field MARL | Phương pháp large-scale cạnh tranh |
| [19] | Jiang et al., ICLR 2020 — Graph Convolutional RL (MHA) | Baseline so sánh |
| [3]  | Agarwal et al., ICML 2019 — SHA | Baseline so sánh |
| [27] | Malysheva et al., NIPS 2018 — RGE | Baseline so sánh |

---

*Tổng hợp bởi Claude · Nguồn: arXiv:2503.07397v1*
