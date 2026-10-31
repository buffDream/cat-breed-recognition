"""录制Windows桌面程序的实际运行；需要requirements-extra.txt中的ffmpeg。"""
import argparse
from pathlib import Path
import subprocess
import sys
import time

import imageio_ffmpeg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('运行演示.mp4'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    output=args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    app=subprocess.Popen([sys.executable,'-X','utf8',str(root/'demo.py'),'--auto-demo'],cwd=root)
    try:
        time.sleep(1)
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-y','-f','gdigrab','-framerate','12',
                 '-draw_mouse','0','-video_size','1080x760','-i','title=Cat Breed Recognition',
                 '-t','72','-c:v','libx264','-preset','veryfast','-threads','2','-crf','21',
                 '-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
        with (root/'artifacts/recording.log').open('w',encoding='utf-8') as log:
            for attempt in range(12):
                result=subprocess.run(command,stdout=log,stderr=log)
                if result.returncode==0:
                    break
                if app.poll() is not None:
                    raise RuntimeError('演示程序提前退出，请查看录制日志。')
                time.sleep(0.5)
            else:
                raise RuntimeError('没有成功录制演示窗口。')
        app.wait(timeout=20)
    finally:
        if app.poll() is None:
            app.terminate()
    print(output)


if __name__=='__main__':
    main()
