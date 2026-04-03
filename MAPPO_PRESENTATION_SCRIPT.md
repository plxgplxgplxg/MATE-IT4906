# Loi dan thuyet trinh: Cach project nay trien khai MAPPO

## 1. Mo dau

Hom nay em se trinh bay cach project nay trien khai thuat toan MAPPO de huan luyen nhieu camera cung phoi hop trong moi truong MATE.

Noi don gian, day la bai toan hoc tang cuong da tac tu. Thay vi chi co 1 tac tu ra quyet dinh, o day co nhieu camera cung hoat dong trong cung mot moi truong, cung quan sat, cung ra hanh dong va cung anh huong den ket qua chung.

Muc tieu cua phan cai dat nay la giup cac camera hoc duoc cach phoi hop tot hon theo thoi gian.

## 2. MAPPO la gi

MAPPO la ten cua thuat toan duoc dung trong project nay.

Co the hieu rat ngan gon nhu sau:

- day la mot thuat toan hoc tang cuong cho nhieu tac tu
- no duoc xay dung du tren y tuong cua PPO de cap nhat chinh sach mot cach on dinh hon

Diem quan trong nhat cua MAPPO la:

- actor hoat dong theo kieu phan tan
- critic hoat dong theo kieu tap trung

Noi de hieu:

- moi camera tu nhin phan thong tin cua chinh no de chon hanh dong
- nhung khi danh gia mot trang thai tot hay xau, critic se nhin thong tin rong hon cua toan he thong

Y tuong nay giup viec hoc on dinh hon, vi trong bai toan nhieu tac tu, neu chi nhin thong tin cuc bo thi rat kho danh gia chinh xac.

## 3. Vi sao project nay chon MAPPO

Project nay mo phong bai toan nhieu camera trong moi truong MATE.

Day la bai toan rat hop voi MAPPO vi:

- co nhieu camera cung ton tai
- cac camera can phoi hop voi nhau
- moi camera chi nhin thay mot phan thong tin
- nhung khi hoc thi can mot cach danh gia mang tinh toan cuc hon

Cho nen MAPPO la lua chon hop ly vi no vua giu duoc viec ra quyet dinh rieng cua tung camera, vua tan dung duoc thong tin chung o phia critic.

## 4. Cau truc cai dat trong project

Trong project nay, phan MAPPO nam chu yeu o cac file sau:

- [gym_agent/algos/on_policy/mappo.py](/Volumes/plxg2/Project/MATE-IT4906/gym_agent/algos/on_policy/mappo.py)
- [gym_agent/algos/on_policy/mappo_core/agent.py](/Volumes/plxg2/Project/MATE-IT4906/gym_agent/algos/on_policy/mappo_core/agent.py)
- [gym_agent/algos/on_policy/mappo_core/networks.py](/Volumes/plxg2/Project/MATE-IT4906/gym_agent/algos/on_policy/mappo_core/networks.py)
- [gym_agent/algos/on_policy/mappo_core/buffer.py](/Volumes/plxg2/Project/MATE-IT4906/gym_agent/algos/on_policy/mappo_core/buffer.py)
- [gym_agent/algos/on_policy/mappo_core/env.py](/Volumes/plxg2/Project/MATE-IT4906/gym_agent/algos/on_policy/mappo_core/env.py)
- [gym_agent/algos/on_policy/mappo_core/state.py](/Volumes/plxg2/Project/MATE-IT4906/gym_agent/algos/on_policy/mappo_core/state.py)

Co the hieu chuc nang tung file nhu sau:

- `mappo.py`: dua thuat toan MAPPO ra ngoai de goi
- `agent.py`: file trung tam, dieu phoi toan bo qua trinh huan luyen
- `networks.py`: dinh nghia mang actor va critic
- `buffer.py`: luu du lieu tam thoi de hoc
- `env.py`: tao nhieu moi truong chay song song
- `state.py`: tao trang thai toan cuc cho critic

Neu phai nho ngan gon thi cu nho:

- `agent.py` la bo nao dieu phoi
- `networks.py` la cac mang hoc
- `buffer.py` la noi luu du lieu
- `env.py` la cau noi voi moi truong

## 5. Y tuong tong the cua luong huan luyen

Luong huan luyen trong project nay co the chia thanh 2 giai doan:

1. Thu thap du lieu tu moi truong
2. Dung du lieu do de cap nhat mo hinh

Noi don gian hon:

- tac tu vao moi truong de chay thu
- trong luc chay thi luu lai quan sat, hanh dong, phan thuong, gia tri du doan va xac suat hanh dong
- sau mot doan du lieu thi lay toan bo du lieu do ra hoc

Day chinh la cach cac thuat toan hoc tang cuong kieu nay thuong hoat dong.

## 6. Project nay khoi tao MAPPO nhu the nao

Khi tao class `MAPPO`, code se lam cac viec sau:

### 6.1. Doc cau hinh

Cau hinh chua cac tham so nhu:

- so moi truong chay song song
- do dai moi dot thu thap du lieu
- toc do hoc
- so lan hoc lai tren cung mot dot du lieu
- kich thuoc bo nho ngan han

Dieu nay giup de chinh tham so ma khong can sua logic chinh.

### 6.2. Tao moi truong MATE

Code dung MATE de tao moi truong nhieu camera, dong thoi dung tac tu target mac dinh la `GreedyTargetAgent`.

Noi de hieu:

- ben camera la phia minh huan luyen
- ben target dang duoc dieu khien boi mot tac tu co luat san

### 6.3. Xac dinh kich thuoc dau vao va dau ra

Tu moi truong, code lay ra:

- co bao nhieu camera
- moi camera quan sat bao nhieu chieu
- hanh dong cua camera co bao nhieu chieu

Day la thong tin dau vao bat buoc de tao actor va critic.

### 6.4. Tao actor va critic

Actor dung de sinh hanh dong.

Critic dung de uoc luong gia tri cua trang thai.

Trong code nay, ca actor va critic deu la mang co bo nho ngan han, tuc la co kha nang giu lai mot phan thong tin cua cac buoc truoc.

Dieu nay phu hop voi bai toan theo thoi gian, vi quyet dinh hien tai thuong phu thuoc vao nhung gi vua xay ra.

## 7. Actor va critic trong project nay hoat dong ra sao

Day la y rat quan trong de thuyet trinh.

### 7.1. Actor

Actor nhan quan sat cua tung camera va sinh ra hanh dong cho camera do.

Trong project nay, actor la `RecurrentGaussianActor`, nghia la:

- co phan bo nho ngan han de nho lich su gan day
- dau ra la mot phan phoi xac suat
- tu phan phoi do se lay ra hanh dong lien tuc

Noi de hieu:

actor khong chi noi "chon hanh dong nao", ma no tao ra mot cach phan bo kha nang cho hanh dong, roi lay hanh dong tu do de tac tu kham pha moi truong.

### 7.2. Critic

Critic khong truc tiep chon hanh dong.

Nhiem vu cua critic la danh gia trang thai hien tai tot hay xau, hay noi cach khac la uoc luong tong loi ich co the nhan duoc trong tuong lai.

Diem dac biet cua MAPPO la critic duoc nhin thong tin rong hon actor. Trong project nay, critic dung `global state` duoc tao boi `state_builder`.

Dieu do giup critic danh gia chinh xac hon trong boi canh nhieu camera cung tuong tac.

## 8. Giai doan 1: Thu thap du lieu

Phan nay nam chu yeu trong ham `collect_rollout()` o [agent.py](/Volumes/plxg2/Project/MATE-IT4906/gym_agent/algos/on_policy/mappo_core/agent.py).

Em co the trinh bay theo tung buoc nhu sau:

### 8.1. Xoa bo nho tam cu

Moi lan bat dau mot dot du lieu moi, bo nho tam cu se duoc xoa de chuan bi luu du lieu moi.

### 8.2. Tao trang thai toan cuc

Tu quan sat hien tai cua tat ca camera, code tao ra `global state` de critic su dung.

Day la diem the hien ro nhat tu tuong critic tap trung.

### 8.3. Actor sinh hanh dong

Actor nhan:

- quan sat hien tai
- bo nho hien tai
- co danh dau day co phai la dau van moi hay khong

Sau do actor tra ve:

- hanh dong
- xac suat log cua hanh dong
- bo nho moi

### 8.4. Critic du doan gia tri

Critic nhan trang thai toan cuc va bo nho cua critic de du doan gia tri.

Gia tri nay rat quan trong o buoc tinh muc loi va tong loi ich sau do.

### 8.5. Dua hanh dong vao moi truong

Sau khi co hanh dong, code goi `env.step(actions)` de lay:

- quan sat moi
- phan thuong
- da ket thuc hay chua
- thong tin van moi

### 8.6. Luu toan bo du lieu vao bo nho tam

Moi buoc thoi gian, code luu lai:

- quan sat
- trang thai toan cuc
- hanh dong
- phan thuong
- da ket thuc hay chua
- gia tri du doan
- xac suat log
- bo nho

Day chinh la du lieu dung de hoc o giai doan sau.

### 8.7. Tinh GAE va return

Sau khi ket thuc mot dot du lieu, code dung critic de lay gia tri o buoc cuoi, roi tinh:

- advantage
- return

Noi de hieu:

- `advantage` cho biet hanh dong vua roi tot hon hay xau hon muc ky vong
- `return` la tong loi ich huong ve tuong lai

Project nay dung GAE de tinh `advantage`.

Noi don gian, cach nay giup viec hoc on dinh hon va bot nhieu nhieu hon.

## 9. Giai doan 2: Cap nhat mo hinh

Phan nay nam trong ham `update()`.

Day la luc MAPPO that su hoc tu du lieu da thu thap.

### 9.1. Chia du lieu thanh cac doan ngan

Vi actor va critic co bo nho ngan han, nen du lieu khong the cat tuy y tung dong mot. Code phai chia thanh cac doan nho theo chuoi thoi gian.

Viec nay giup mang hoc dung theo thu tu thoi gian.

### 9.2. Tinh lai xac suat moi

Code chay lai actor tren du lieu cu de tinh `new_log_probs`, roi so voi `old_log_probs` da luu khi thu thap du lieu.

Tu do co the biet chinh sach moi da thay doi bao nhieu.

### 9.3. Tinh sai so actor theo PPO

Day la phan quan trong nhat cua PPO.

Y tuong la:

- neu chinh sach moi tot hon thi khuyen khich cap nhat
- nhung neu thay doi qua manh thi phai chan lai

Nho vay qua trinh hoc on dinh hon.

Neu can noi ngan gon khi thuyet trinh, co the noi:

"Actor duoc cap nhat theo PPO de cai thien cach ra hanh dong, nhung van tranh thay doi qua dot ngot."

### 9.4. Tinh sai so critic

Critic hoc cach du doan `return`.

Trong code nay, phan sai so cua critic co them co che chan va dung Huber loss de hoc on dinh hon.

Noi don gian:

- critic co gang du doan dung gia tri tuong lai
- nhung cung bi kiem soat de khong cap nhat qua manh

### 9.5. Lan truyen nguoc

Sau khi co sai so cua actor va critic, code:

- cong hai sai so lai
- goi lan truyen nguoc
- gioi han do lon gradient
- cap nhat tham so bang Adam

Day la buoc mo hinh hoc that su tu du lieu.

## 10. Vai tro cua buffer trong project nay

Buffer la thanh phan rat quan trong vi no giu toan bo du lieu trung gian giua luc tac tu tuong tac voi moi truong va luc mo hinh hoc.

Neu khong co buffer thi khong the:

- tinh advantage
- tinh return
- chia thanh tung nhom de hoc
- hoc lai nhieu lan tren cung mot dot du lieu

Noi ngan gon:

buffer la noi gom du lieu de bien trai nghiem chay thu thanh du lieu huan luyen.

## 11. Vi sao project nay dung mang co bo nho

Day la diem nen nhan manh khi thuyet trinh.

Trong bai toan camera theo doi muc tieu, quan sat o mot thoi diem co the khong du de hieu toan bo tinh huong.

Vi du:

- muc tieu vua bien mat khoi vung nhin
- camera can dua vao nhung gi vua xay ra truoc do de ra quyet dinh tiep theo

Vi vay project nay dung actor va critic co bo nho de giu lai mot phan ky uc ngan han.

Dieu nay giup camera ra quyet dinh tot hon trong moi truong co tinh thoi gian.

## 12. Diem nao trong project nay the hien dung tinh than MAPPO

Co 4 y quan trong:

### 12.1. Nhieu tac tu cung hoc

Moi camera la mot tac tu.

### 12.2. Dung chung mot chinh sach

Tat ca camera dung chung cung mot actor network.

Dieu nay giup giam so luong tham so va tan dung du lieu tu nhieu camera de hoc nhanh hon.

### 12.3. Critic tap trung

Critic khong chi nhin thong tin cua mot camera, ma nhin trang thai toan cuc rong hon.

Day la dac trung rat quan trong cua MAPPO.

### 12.4. Cap nhat kieu PPO

Qua trinh cap nhat dung co che cua PPO de giu tinh on dinh.

## 13. Neu thay co hoi "project nay da cai dat MAPPO o dau"

Co the tra loi ngan gon nhu sau:

"Phan cai dat MAPPO cua project nam chu yeu trong `mappo_core`. Trong do `agent.py` dieu phoi vong lap huan luyen, `networks.py` dinh nghia actor va critic co bo nho, `buffer.py` luu du lieu de tinh advantage va cap nhat theo PPO, con `env.py` va `state.py` giup noi thuat toan voi moi truong MATE."

## 14. Neu thay co hoi "khac gi PPO thuong"

Co the tra loi:

"Khac biet chinh la o day khong con mot tac tu duy nhat. Bai toan co nhieu camera cung hoat dong. Vi vay project dung MAPPO, tuc la mo rong PPO sang boi canh nhieu tac tu. Actor van ra quyet dinh rieng cho tung camera, nhung critic duoc cap thong tin tong quat hon de danh gia trang thai tot hon."

## 15. Neu thay co hoi "vi sao khong dung critic rieng cho tung agent"

Co the tra loi:

"Trong moi truong nhieu tac tu, neu moi critic chi nhin thong tin cuc bo thi viec danh gia trang thai se thieu chinh xac. Critic tap trung giup qua trinh hoc on dinh hon vi no nhin duoc boi canh rong hon cua toan he thong."

## 16. Neu thay co hoi "vi sao can hidden state"

Co the tra loi:

"Vi day la moi truong co tinh chuoi thoi gian. Quan sat tai mot thoi diem khong phai luc nao cung du. Bo nho ngan han giup actor va critic giu thong tin tu qua khu gan, nen viec ra quyet dinh va uoc luong gia tri tot hon."

## 17. Ket luan ngan gon

Tom lai, project nay trien khai MAPPO theo huong rat ro rang:

- dung nhieu moi truong de thu thap du lieu song song
- moi camera la mot tac tu
- actor co bo nho sinh hanh dong tu quan sat cuc bo
- critic co bo nho danh gia gia tri tu trang thai toan cuc
- dung buffer de luu du lieu
- dung GAE de tinh advantage
- dung PPO de cap nhat on dinh

Noi ngan gon trong mot cau:

"Project nay da dua MAPPO vao MATE bang cach ket hop mot chinh sach chung cho nhieu camera, critic tap trung de danh gia trang thai toan cuc, va co che cap nhat on dinh cua PPO."

## 18. Ban cuc ngan de noi trong 30 giay

"Trong project nay, em trien khai MAPPO de huan luyen nhieu camera trong moi truong MATE. Moi camera dung actor de chon hanh dong tu quan sat cua rieng no, con critic dung trang thai toan cuc de danh gia tot hon toan bo tinh huong. Du lieu duoc luu vao buffer, sau do em dung GAE va PPO de cap nhat mo hinh mot cach on dinh."

## 19. Ban de nho de noi trong 1 phut

"Y tuong chinh cua phan cai dat nay la chuyen bai toan camera theo doi muc tieu thanh bai toan hoc tang cuong da tac tu. Moi camera la mot tac tu. Actor cua moi camera chon hanh dong dua tren quan sat cuc bo, nhung critic thi nhin thong tin rong hon de danh gia trang thai. Trong luc chay, he thong thu thap du lieu tu nhieu moi truong song song, luu vao buffer, tinh advantage bang GAE, roi cap nhat actor va critic bang co che cua PPO. Vi vay day la mot cai dat MAPPO phu hop voi bai toan nhieu camera can phoi hop voi nhau."
