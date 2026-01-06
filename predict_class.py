from torchvision import io as tvio 
from torchvision import models
import torchinfo


# 画像を読み込む
input_image = tvio.decode_image('assets/10Tank.jpg')
print(type(input_image))
print(input_image.shape, input_image.dtype)

# 学習済みモデルの重みを読み込む
weights = models.GoogLeNet_Weights.DEFAULT

# モデルを作る
model = models.googlenet(weights=weights)
# print(model)
torchinfo.summary(model)

# モデル前処理の方法を取得する
preprocess = weights.transforms()

# バッチにする
batch = preprocess(input_image).unsqueeze(dim=0)
print(batch.shape)

# モデルを推論モードにする
model.eval()

# バッチに対する推論
output_logits = model(batch)
print(output_logits.shape, output_logits.dtype)

# バッチ内のデータごとにクラス確率に変換する
output_probs = output_logits.softmax(dim=1)

# バッチからインデックス 0 のデータを取り出して、
# 結果を表示する
class_id = output_probs[0].argmax().item()
score = output_probs[0][class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")