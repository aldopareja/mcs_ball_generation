import subprocess as sp
import multiprocessing as mp
from pathlib import Path
from time import sleep

def spawn_forever(r, save_data_folder, scenes_json_folder):
  while True:
    try:
      _ = sp.check_call(["python", "generate_data.py", f'{r}', save_data_folder, scenes_json_folder], stderr=sp.STDOUT)
    except sp.CalledProcessError:
      sp.check_call(["h5clear", "-s", str(Path(save_data_folder) / f"soccer_balls_data_{r}.h5")])
      print("again", r)

if __name__ == "__main__":
  ps = []
  sp.check_call(["which", "python"])
  for r in range(40):
    p = mp.Process(target=spawn_forever, args=(r, "dist_data/", "ball_scenes/"))
    p.start()
    ps.append(p)
    sleep(0.5)
  for p in ps:
    p.join()