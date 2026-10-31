"""由用户手动运行：把本地已有Git提交推送到一个空的Gitee仓库。"""
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlparse


def repository_url(value):
    url = urlparse(value.strip())
    parts = url.path.strip('/').removesuffix('.git').split('/')
    if (url.scheme != 'https' or url.hostname != 'gitee.com' or url.username or url.password
            or url.port not in (None, 443) or url.query or url.params or url.fragment or len(parts) != 2
            or not all(p not in ('.', '..') and re.fullmatch(r'[A-Za-z0-9_.-]+', p) for p in parts)):
        raise ValueError('请粘贴 https://gitee.com/用户名/仓库名 这样的仓库地址。')
    return f'https://gitee.com/{parts[0]}/{parts[1]}'


def main():
    root = Path(__file__).resolve().parent
    git = shutil.which('git')
    if not git:
        bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
        if bundled.is_file():
            git = str(bundled)
    if not git:
        raise RuntimeError('没有找到Git。请安装Git for Windows后重新打开本程序。')
    if not (root / '.git').is_dir():
        raise RuntimeError('此目录没有开发历史。请在原项目目录或包含.git的源码包中运行。')
    print('此程序会上传本项目已有的Git提交。模型权重和视频另交给同学，不随源码推送。')
    url = repository_url(input('粘贴刚创建的空Gitee仓库地址，然后回车：'))
    heads = subprocess.run([git, 'ls-remote', '--heads', url + '.git'], cwd=root,
                           text=True, capture_output=True)
    if heads.returncode:
        raise RuntimeError('仓库访问失败，请在Gitee完成登录，并确认仓库地址正确。\n' + heads.stderr)
    if heads.stdout.strip():
        raise RuntimeError('该仓库已有分支。请新建空仓库，不勾选初始化README，再运行本程序。')
    subprocess.run([git, 'push', url + '.git', 'HEAD:refs/heads/main'], cwd=root, check=True)
    (root / '仓库链接.txt').write_text(url + '\n', encoding='utf-8')
    print('\n上传完成。你的仓库链接：' + url)
    print('打开仓库网页确认能看到README.md，再把这个链接和完整成果包一起发给考核群缪延瑞同学。')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print('\n未完成上传：' + str(error))
    input('\n按回车关闭。')
