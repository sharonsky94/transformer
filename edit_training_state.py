import os
import numpy as np

nn_file = 'output/sl8_b500_as1/model_checkpoint_0.98.keras' # путь к файлу модели
state_file = 'training_state.npy'

if os.path.exists(state_file):
    create_nn = False
    state = np.load(state_file, allow_pickle=True).item()
    start_epoch = state.get('epoch', 0)
    start_step = state.get('step', 0)
    saved_step_index = state.get('feistel_index', 0)
    nn_file = state.get('nn_file', 0)
    print(f"🔁 Восстановление обучения: эпоха {start_epoch}, шаг {start_step}, индекс {saved_step_index}, модель {nn_file}")
else:
    create_nn = True
    start_epoch = 0
    start_step = 0
    saved_step_index = 0


    
# Сохраняем состояние после эпохи
state = {
    'epoch': 0,
    'step': 0,
    'feistel_index': 0,
    'nn_file': nn_file
}
#np.save(state_file, state)