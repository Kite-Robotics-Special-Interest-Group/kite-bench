# kite-bench

> VLA 鐨勪豢鐪熺幆澧冨拰杈撳叆椴佹鎬ц瘎娴嬪彴銆傜墿鐞嗙敤 MuJoCo锛屽瑙傛壈鍔ㄥ彲浠ユ帴 Unity / Moer Engine銆?
涓€涓?VLA 妯″瀷鍦ㄨ缁冨垎甯冮噷鑳芥姄浣忔柟鍧楋紝涓嶄唬琛ㄥ畠鍦ㄧ湡瀹炶溅闂撮噷杩樿兘鎶撲綇鈥斺€旀崲涓€鐩忕伅銆佺浉鏈鸿纰版 5掳銆佺洰鏍囪鏈烘鑷傝嚜宸辨尅浣忥紝杈撳嚭灏卞彲鑳藉畬鍏ㄨ窇鍋忋€?*kite-bench 灏辨槸涓鸿繖浠朵簨鍑嗗鐨?*锛氱粰鍚屼竴鎵硅娴嬪拰鍚屼竴鎵?episode 鍔犱笂鍙帶鎵板姩锛岄噺鍖栨ā鍨嬬殑鍔ㄤ綔婕傜Щ鍜屾垚鍔熺巼涓嬮檷銆?
## 涓轰粈涔堥渶瑕佸畠

- 鐜版湁 VLA 璁烘枃澶у鍙姤鍒嗗竷鍐呮垚鍔熺巼锛屾壈鍔ㄤ笅鐨勯€€鍖栨病鏈夌粺涓€鍙ｅ緞
- 鐪熷疄閮ㄧ讲鐨勭涓€涓棶棰樹笉鏄?鑳戒笉鑳藉仛"锛岃€屾槸"杈撳叆鍙樹竴鐐硅繕鑳戒笉鑳藉仛"
- 娑堣垂绾ф樉鍗★紙8 GB锛変篃搴旇鑳借窇瀹屾暣璇勬祴

## 涓ゅ眰璇勬祴

| 灞?| 鎵板姩鏉ユ簮 | 宸ュ叿 | 鎴愭湰 |
| --- | --- | --- | --- |
| 杈撳叆灞?| 鍏夊害銆佸嚑浣曘€侀伄鎸°€佽瑷€銆佹椂搴?| 绾?Python锛圥IL/numpy锛?| 浣庯紝鍙窇涓婂崈缁?|
| 澶栬灞?| 鍏夌収銆佹潗璐ㄣ€佽儗鏅€佺浉鏈烘満浣嶃€侀伄鎸＄墿 | Unity / Moer Engine 閲嶆覆鏌?| 涓紝鍑犲崄缁?|
| 闂幆灞?| 涓婅堪鎵板姩涓嬬殑瀹屾暣 episode | MuJoCo锛堢墿鐞嗘纭€佸彲鎵归噺锛?| 涓紝姣忕被 20鈥?0 episode |

鍔ㄤ綔婕傜Щ鏄究瀹滅殑浠ｇ悊鎸囨爣锛岀敤鏉ユ壂鍙傛暟锛?*缁撹绾ф寚鏍囨槸闂幆鎴愬姛鐜?*銆?
## 鎸囨爣

- `螖pos`锛氭湯绔綅缃樊鐨?L2锛堟绫筹級
- `螖rot`锛氬Э鎬佹祴鍦拌窛绂伙紙搴︼級
- `grip_flip`锛氬す鐖紑鍚堝垽瀹氱炕杞巼
- `success@perturb`锛氭壈鍔ㄤ笅 episode 鎴愬姛鐜囷紝涓庡共鍑€鍩虹嚎瀵规瘮
- `robust_score = 1 - mean(螖pos / scale)`锛堝綊涓€鍖栨紓绉伙紝瓒婂ぇ瓒婄ǔ锛?
鍏紡涓庡彛寰勮 [`docs/design.md`](docs/design.md)銆?
## 鐩綍

```
src/kitebench/
  perturb.py     # 鍏夊害/鍑犱綍/閬尅鎵板姩锛堢函 PIL锛屾棤浠跨湡渚濊禆锛?  metrics.py     # 婕傜Щ涓庢垚鍔熺巼鎸囨爣
docs/
  design.md      # 璁捐涓庢寚鏍囧彛寰?  roadmap.md     # 閲岀▼纰?tests/
```

## 蹇€熷紑濮?
```bash
pip install -e ".[sim,dev]"   # sim 额外装 mujoco
python -m pytest -q
```

MuJoCo 的 WidowX-250 模型不随仓库分发：首次构建场景时会从 `mujoco_menagerie` 稀疏克隆到 `~/.cache/kitebench/models`，也可以用 `KITEBENCH_WX250S` 指向已有目录。没有 mujoco 时仿真测试自动跳过，CI 只跑不依赖仿真的部分。

```python
from PIL import Image
from kitebench import perturb, metrics

img = Image.open("frame.png")
for name, fn in perturb.registry().items():
    out = fn(img, seed=0)
    print(name, out.size)
```

## 涓?Kite 鐨勫叧绯?
- `Kite`锛氱渚?VLA 妯″瀷涓庤繍琛屾椂锛堟満鍣ㄤ汉鏈綋锛?- `kite-bench`锛氳瘎娴嬪彴锛?*涓嶅寘鍚ā鍨嬫潈閲?*锛屼篃涓嶄緷璧栧巶鍟嗗伐鍏烽摼
- 涓よ€呴€氳繃"瑙傛祴 鈫?鍔ㄤ綔"鐨勬帴鍙ｈВ鑰︼細浠讳綍鑳藉悙 30脳20 鍔ㄤ綔鍧楃殑绛栫暐閮借兘鎺ヨ繘鏉?
## 璁稿彲

MIT锛岃 [`LICENSE`](LICENSE)銆?
