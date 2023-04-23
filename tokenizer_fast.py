########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys
import subprocess

dirname = os.path.dirname(__file__)
release_dir = os.path.join(dirname, "target", "release")

if not (os.path.exists(os.path.join(release_dir, "librwkv_tokenizer.so")) or \
        os.path.exists(os.path.join(release_dir, "librwkv_tokenizer.dll")) or \
        os.path.exists(os.path.join(release_dir, "librwkv_tokenizer.dylib"))):

    result = subprocess.run(["cargo", "--version"], stdout=subprocess.DEVNULL)
    if result.returncode != 0:
        print("Rust is not installed; go to https://rustup.rs to install it or install it with the following command:")
        print("  $ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        exit(1)

    status = os.system("cargo build --release")
    if status != 0:
        print("Failed to compile the native extension!")
        exit(1)

sys.path.append(release_dir)

from librwkv_tokenizer import encode, decode
from tokenizer_original import RWKV_TOKENIZER

TOKENIZER = RWKV_TOKENIZER('rwkv_vocab_v20230422.txt')

src = 'Õ\U000683b8'
assert encode(src) == TOKENIZER.encode(src)
assert decode(encode(src)) == src

src = '''起業家イーロン・マスク氏が創業した宇宙開発企業「スペースX（エックス）」の巨大新型ロケット「スターシップ」が20日朝、初めて打ち上げられたが、爆発した。
打ち上げは米テキサス州の東海岸で行われた。無人の試験で、負傷者はいなかった。
打ち上げから2～3分後、史上最大のロケットが制御不能になり、まもなく搭載された装置で破壊された。
マスク氏は、数カ月後に再挑戦すると表明した。
スペースXのエンジニアたちは、それでもこの日のミッションは成功だったとしている。「早期に頻繁に試験する」ことを好む人たちなので、破壊を恐れていない。次のフライトに向け、大量のデータを収集したはずだ。2機目のスターシップは、ほぼ飛行準備が整っている。
マスク氏は、「SpaceXチームの皆さん、スターシップのエキサイティングな試験打ち上げ、おめでとう！　数カ月後に行われる次の試験打ち上げに向けて、多くを学んだ」とツイートした。
アメリカでのロケット打ち上げを認可する米連邦航空局（NASA）は、事故調査を監督するとした。広報担当者は、飛行中に機体が失われた場合の通常の対応だと述べた。
マスク氏は打ち上げ前、期待値を下げようとしていた。発射台の設備を破壊せずに機体を打ち上げるだけでも「成功」だとしていた。
その願いはかなった。スターシップは打ち上げ施設からどんどん上昇し、メキシコ湾の上空へと向かっていった。しかし1分もしないうち、すべてが計画通りに進んでいるのではないことが明らかになった。'''

assert TOKENIZER.encode(src) == encode(src)
assert decode(encode(src)) == src

import time

start = time.time()
for _ in range(2000):
    TOKENIZER.encode(src)
end = time.time()
print("Original tokenizer: {}us".format(int((end - start) * 1000000)))


start = time.time()
for _ in range(2000):
    encode(src)
end = time.time()
print("Fast tokenizer: {}us".format(int((end - start) * 1000000)))
