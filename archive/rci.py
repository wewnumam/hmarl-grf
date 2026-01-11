import numpy as np
import gfootball.env as football_env
import types
# Import modul 'action_set' untuk mendefinisikan aksi ideal secara eksplisit
import gfootball.env.football_action_set as action_set

# === PATCH: GRF Compatibility for Gymnasium / NumPy 2.x ===
def patch_grf_env(env):
    """Patch GRF env to handle Gymnasium-style API calls safely."""
    orig_reset = env.reset
    orig_step = env.step

    def reset_wrapper(self, *args, **kwargs):
        # Accept all Gymnasium args (seed, options, etc.)
        try:
            result = orig_reset(*args, **kwargs)
        except TypeError:
            # fallback if old signature causes error
            result = orig_reset()
        if isinstance(result, tuple):
            return result
        return result, {}

    def step_wrapper(self, *args, **kwargs):
        result = orig_step(*args, **kwargs)
        if len(result) == 5:  # Gymnasium format
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        else:  # GRF / old Gym format
            return result

    env.reset = types.MethodType(reset_wrapper, env)
    env.step = types.MethodType(step_wrapper, env)
    return env
# ==========================================================


# === FUNGSI LOGIKA RCI ===

def sim_binary(actual_action, ideal_role_action):
    """
    (Persamaan 8/9, varian biner)
    Mengukur kesesuaian: 1 jika sama, 0 jika berbeda.
    
    PERBAIKAN: Melakukan cast 'ideal_role_action' ke int() untuk 
    menghindari type mismatch. 'actual_action' adalah int, 
    sedangkan 'ideal_role_action' adalah objek Enum/Action.
    """
    # Membandingkan int == int(ObjekAksi)
    try:
        # Mengubah objek aksi (misal: action_set.action_dribble) menjadi integer
        ideal_action_as_int = int(ideal_role_action)
        return 1 if (actual_action == ideal_action_as_int) else 0
    except Exception as e:
        # Menangani jika ada error saat casting
        print(f"Error casting action: {e}. ideal_role_action was: {ideal_role_action}")
        return 0

def get_ideal_role_action(obs_dict, player_index, gk_idx, def_indices, 
                          player_pos, formation_pos, ball_pos, move_fn):
    """
    Model Hierarki Taktis Ideal (r_i)
    Ini adalah "kunci jawaban" peran Anda.
    
    PERBAIKAN: Menggunakan nama atribut yang benar (misal: action_dribble)
    """
    role = "other"
    if player_index == gk_idx:
        role = "gk"
    elif player_index in def_indices:
        role = "defender"

    left_has_ball = obs_dict['ball_owned_team'] == 0
    ball_owner = obs_dict['ball_owned_player'] if left_has_ball else None

    # 1. Logika Ideal SAAT PUNYA BOLA
    if left_has_ball and ball_owner == player_index:
        if role == "gk":
            # PERBAIKAN: 'action_long_pass'
            return action_set.action_long_pass # GK buang bola jauh
        
        # Jika pemain berada di area serang (misal x > 0.7)
        if player_pos[0] > 0.7:
            # PERBAIKAN: 'action_shot'
            return action_set.action_shot
        else:
            # PERBAIKAN: 'action_dribble'
            return action_set.action_dribble # Bawa bola ke depan

    # 2. Logika Ideal SAAT TIDAK PUNYA BOLA
    else:
        # Untuk GK dan Bek, peran ideal adalah KEMBALI KE FORMASI
        if role in ["gk", "defender"]:
            return move_fn(player_pos, formation_pos)
        # Untuk Gelandang/Penyerang, peran ideal adalah MENGEJAR BOLA
        else:
            return move_fn(player_pos, ball_pos)
# ============================


# === Create patched environment ===
env = football_env.create_environment(
    env_name="11_vs_11_stochastic",
    representation="raw",
    render=False, # Set ke True jika ingin melihat visual
    number_of_left_players_agent_controls=11
)
env = patch_grf_env(env)

# === Formation logic etc. ===
formation = [
    (0.05, 0.50),  # GK
    (0.15, 0.35), (0.15, 0.40), (0.15, 0.60), (0.15, 0.65),
    (0.35, 0.20), (0.35, 0.40), (0.35, 0.60), (0.35, 0.80),
    (0.55, 0.40), (0.55, 0.60)
]

GK_INDEX = 0
BACK4_INDICES = [1, 2, 3, 4]
FPS = 30
PASS_DELAY = 3 * FPS
possession_timer = np.zeros(11, dtype=int)

# PERBAIKAN: Mengklarifikasi konstanta aksi berdasarkan dokumentasi
# Aksi '9' di action set default adalah 'long_pass'
ACTION_LONG_PASS = 9    # action_set.action_long_pass
ACTION_SHORT_PASS = 11   # action_set.action_short_pass
ACTION_SPRINT = 13       # action_set.action_sprint


def move_towards(player_pos, target_pos):
    dx, dy = target_pos[0] - player_pos[0], target_pos[1] - player_pos[1]
    if abs(dx) < 0.02 and abs(dy) < 0.02: return 0 # idle
    if dx > 0 and abs(dy) < 0.1: return 5 # right
    if dx < 0 and abs(dy) < 0.1: return 1 # left
    if dy > 0 and abs(dx) < 0.1: return 7 # bottom
    if dy < 0 and abs(dx) < 0.1: return 3 # top
    if dx > 0 and dy > 0: return 6 # bottom_right
    if dx > 0 and dy < 0: return 4 # top_right
    if dx < 0 and dy > 0: return 8 # bottom_left
    if dx < 0 and dy < 0: return 2 # top_left
    return 0 # idle

# === Main loop ===
# PERBAIKAN: Menggunakan API reset yang benar (mengembalikan obs dan info)
obs, info = env.reset()
done, step = False, 0

# Inisialisasi pelacak RCI
total_sim_score = 0
total_agent_decisions = 0

while not done:
    step += 1
    # Dalam mode 'raw' multi-agen, obs adalah list.
    # Kita ambil obs[0] karena semua obs agen berisi info global
    obs_dict = obs[0] if isinstance(obs, list) else obs

    player_positions = np.array(obs_dict['left_team'])
    norm_positions = (player_positions + 1) / 2.0
    ball_pos = (np.array(obs_dict['ball'][:2]) + 1) / 2.0

    left_has_ball = obs_dict['ball_owned_team'] == 0
    ball_owner = obs_dict['ball_owned_player'] if left_has_ball else None

    actual_actions_ai = [] # List untuk aksi aktual (a_i)
    
    for i in range(11):
        pos = norm_positions[i]

        # === LOGIKA AGEN ANDA (a_i) ===
        # (Logika ini adalah 'apa yang sebenarnya dilakukan' agen)
        if left_has_ball and ball_owner == i:
            possession_timer[i] += 1
            if possession_timer[i] >= PASS_DELAY:
                # PERBAIKAN: Kode Anda menggunakan 9, yang merupakan LONG_PASS
                actual_act = ACTION_LONG_PASS 
                possession_timer[i] = 0
            else:
                # PERBAIKAN: Kode Anda menggunakan 9, yang merupakan LONG_PASS
                actual_act = ACTION_LONG_PASS # Menahan bola
        elif left_has_ball:
            # PERBAIKAN: Kode Anda menggunakan 9, yang merupakan LONG_PASS
            actual_act = ACTION_LONG_PASS
        else:
            if i in BACK4_INDICES or i == GK_INDEX:
                target = formation[i]
                actual_act = move_towards(pos, target)
            else:
                # PERBAIKAN: Kode Anda menggunakan 9, yang merupakan LONG_PASS
                actual_act = ACTION_LONG_PASS
            possession_timer[i] = 0
        
        actual_actions_ai.append(actual_act)
        # === SELESAI LOGIKA AGEN ===


        # === PERHITUNGAN RCI ===
        # 1. Dapatkan Aksi Ideal (r_i) dari Model Taktis
        ideal_act_ri = get_ideal_role_action(
            obs_dict, i, GK_INDEX, BACK4_INDICES,
            pos, formation[i], ball_pos, move_towards
        )

        # 2. Hitung kesesuaian (sim)
        # Ini sekarang memanggil fungsi sim_binary yang sudah diperbaiki
        sim_score = sim_binary(actual_act, ideal_act_ri)
        total_sim_score += sim_score
        # === SELESAI RCI ===

    # Akumulasi total keputusan (11 agen x 1 langkah)
    total_agent_decisions += 11

    # Lingkungan dieksekusi menggunakan aksi AKTUAL (a_i)
    # PERBAIKAN: Menggunakan API step yang dipatch (mengembalikan 4 nilai)
    obs, reward, done, info = env.step(actual_actions_ai)
    
    # env.render() # Aktifkan jika ingin melihat
    
    if step % 100 == 0:
        print(f"Step {step}...")
        
    # Berhenti setelah 500 langkah untuk demo
    if step > 500:
        done = True


env.close()

# === HASIL AKHIR RCI ===
print("\n=========================================")
print(f"   ROLE COHERENCE INDEX (RCI) - EPISODE")
print("=========================================")
if total_agent_decisions > 0:
    episode_RCI = total_sim_score / total_agent_decisions
    print(f"Total Keputusan Agen: {total_agent_decisions}")
    print(f"Total Aksi Koheren: {total_sim_score}")
    print(f"RCI Episode Final:  {episode_RCI:.4f}")
else:
    print("Episode selesai tanpa ada keputusan.")