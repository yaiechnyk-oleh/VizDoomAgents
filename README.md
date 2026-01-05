# VizDoomAgents

RL-агент(и) для **ViZDoom** у сценарії *deathmatch* з фокусом на стабільному навчанні та діагностиці поведінки.

- **Алгоритм:** RecurrentPPO (SB3-Contrib)  
- **Політика:** кастомний **CNN feature extractor + GRU** (див. `cnn_gru.py`)  
- **Середовище:** `DoomDeathmatchEnv` (Gym/Gymnasium-сумісне) з reward shaping та логуванням (див. `env.py`)  
- **Артефакти:** чекпойнти/моделі → `models/`, результати оцінювання → `results/`, агреговані CSV → `data/`

---

## Швидкий старт

### 1) Встановлення залежностей

> **Важливо:** `train.py` та `cnn_gru.py` використовують `sb3-contrib` (RecurrentPPO), тому встановіть його окремо (у `requirements.txt` його може не бути).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install sb3-contrib
```

### 2) Smoke test (перевірка, що середовище працює)

```bash
python smoke_test.py --cfg configs/deathmatch_base.cfg
```

### 3) Тренування

Приклад запуску тренування (рекомендовані параметри можна змінювати під ваше залізо):

```bash
python train.py   --cfg configs/deathmatch_base.cfg   --persona rusher   --seed 42   --n_envs 4   --timesteps 800000   --eval_freq 100000   --eval_episodes 30   --lr 1e-4   --ent_coef 0.02   --ent_final 0.008
```

Під час тренування:
- чекпойнти зберігаються у `models/` (`ckpt_<persona>_<runid>_...`)
- “останній” стан моделі → `models/<persona>_last_model.zip`
- “найкращий” (за періодичним eval) → `models/best_<persona>.zip`

### 4) Оцінювання (evaluation)

```bash
python eval.py   --cfg configs/deathmatch_base.cfg   --persona rusher   --model models/rusher_last_model.zip   --episodes 10   --seed 123   --out results/eval_rusher.csv
```

Опції:
- `--watch` — показувати вікно ViZDoom під час оцінювання
- `--stochastic` — оцінювання з `deterministic=False`
- `--diag` — детальна діагностика (частота/перелік керуються `--diag_*`)

---

## Що всередині

### Середовище (`env.py`)

`DoomDeathmatchEnv` реалізує:
- **Спостереження:** зображення (stacked frames) у форматі **CHW** (uint8)
- **Дії:** дискретний список дій, сформований з `available_buttons` у `.cfg`
- **Завершення епізоду:**
  - смерть → `terminated=True`
  - ліміт кроків → `truncated=True`
  - опційно “застрягання” → дострокове завершення з штрафом

Reward shaping (скорочено):
- винагорода за **damage dealt / hitcount**
- штраф за **damage taken / hits taken / death**
- винагорода за “атрибутовані” вбивства монстрів та штраф за frag (за потреби)
- **гігієна стрільби/перемикання зброї** (штрафи за спам/порожні постріли тощо)
- **goal-based shaping**: перемикання цілей (enemy/search/pickup) за критичними умовами HP/AMMO

> За замовчуванням рекомендується **не** використовувати “сирий” reward від `make_action`.  
> Увімкнути можна прапором `--use_game_reward` (але він шумний та залежить від сценарію).

### Політика (`cnn_gru.py`)
- `CustomCNN` — feature extractor для CHW (підтримує будь-яку кількість каналів = кількість stacked frames)
- `CnnGruPolicy` — заміна LSTM в RecurrentActorCriticPolicy на GRU при збереженні інтерфейсу станів

### Колбеки та діагностика (`callback.py`)
У тренуванні використовуються колбеки для:
- **entropy annealing** (`ent_coef -> ent_final`)
- періодичного eval + збереження “best”
- детекції деградації у політиці (action collapse), підозрілих reward-сплесків тощо
- агрегації `info`-метрик зі середовища

---

## Конфіги та сценарій

- `configs/deathmatch_base.cfg` — базовий конфіг (рендеринг, кнопки, timeout тощо)
- `configs/deathmatch_vars.cfg` — варіант з мінімальними game variables (KILLCOUNT/HEALTH/ARMOR/WEAPON)
- `configs/deathmatch.wad` — WAD зі сценарієм

> Якщо ви додаєте власний `.cfg`, переконайтесь, що `doom_scenario_path` вказує на існуючий `.wad` (часто зручно тримати `.cfg` і `.wad` поруч у `configs/`).

---

## Корисні утиліти

### Розкодувати індекси дій у кнопки
```bash
python utils/decode_actions.py --cfg configs/deathmatch_base.cfg
```

---

## Параметри, які часто змінюють

- `--n_envs` — кількість паралельних середовищ
- `--frame_skip` — пропуск кадрів (швидкість vs контроль)
- `--max_steps` — довжина епізоду
- `--disable_weapon_actions` — прибрати actions для вибору зброї (для експериментів)
- `--own_kill_user_var` — якщо у вашому WAD є USERk game variable для “власних” вбивств  
  Також можна задати через env var: `DOOM_OWN_KILL_USER_VAR`

---

## Структура репозиторію

- `train.py` — тренування RecurrentPPO
- `eval.py` — оцінювання та експорт метрик у CSV
- `env.py` — Gym/Gymnasium середовище + reward shaping
- `cnn_gru.py` — CNN + GRU політика для SB3-Contrib
- `callback.py` — колбеки/діагностика/логування
- `configs/` — `.cfg` і `.wad`
- `models/` — збережені моделі та чекпойнти
- `results/` — приклади CSV/логів оцінювання
- `data/` — агреговані датасети (CSV), якщо ви збираєте статистику по тиках/епізодах
- `dataset/` — скрипти для генерації/агрегації табличних даних (для аналізу)

---

## Типові проблеми

- **`ImportError: sb3_contrib ...`** → встановіть `pip install sb3-contrib`
- **ViZDoom не знаходить WAD** → перевірте `doom_scenario_path` у `.cfg` та шлях запуску (cwd)
- **Падає рендер/вікно** → запускайте без `--watch` або з `window_visible = false` у cfg

---

## Ліцензія
Якщо потрібно — додайте файл `LICENSE` та вкажіть тип ліцензії (MIT/Apache-2.0 тощо).
