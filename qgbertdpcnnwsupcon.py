import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, create_optimizer
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ======= 1. 参数设置 =======
train_file = "/root/data/train.tsv"
dev_file = "/root/data/dev.tsv"
test_file = "/root/data/test.tsv"
bert_dir = '/root/bert-base-chinese'
save_model_weights = '/tmp/pycharm_project_10/day01/bert_dpcnn_supcon.weights.h5'
output_dir = '/root/data'
max_length = 128
feature_dim = 256
batch_size = 16
num_blocks = 2
epochs = 5

# 加权方式设置
WEIGHT_TYPE = "dynamic"   # "dynamic" or "static"
alpha_static = 0.25       # 静态加权用（如消融时也可换成0.5等）

# ======= 2. 数据加载 =======
def load_data(file):
    df = pd.read_csv(file, sep="\t")
    texts = df['text_a'].astype(str).tolist()
    labels = df['label'].astype(int).tolist() if 'label' in df.columns else None
    return texts, (np.array(labels) if labels is not None else None), df

texts_train, y_train, _ = load_data(train_file)
texts_val, y_val, _ = load_data(dev_file)
texts_test, y_test, test_df = load_data(test_file)

# ======= 3. 分词编码 =======
tokenizer = BertTokenizer.from_pretrained(bert_dir)
def encode(texts):
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )

def batch_encode(texts):
    enc = encode(texts)
    return {
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'token_type_ids': enc['token_type_ids']
    }

X_train = batch_encode(texts_train)
X_val = batch_encode(texts_val)
X_test = batch_encode(texts_test)

# ======= 4. Mish激活 =======
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
keras.utils.get_custom_objects().update({'mish': keras.layers.Activation(mish)})

# ======= 5. SupConLoss定义 =======
class SupConLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.10, name='supcon_loss'):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, labels, features):
        features = tf.math.l2_normalize(features, axis=1)
        batch_size = tf.shape(features)[0]
        similarity = tf.matmul(features, features, transpose_b=True)
        sim_div = similarity / self.temperature

        labels = tf.reshape(labels, [-1, 1])
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
        logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
        mask *= logits_mask
        easy_weights = mask / (tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-9)
        weights = easy_weights

        exp_sim = tf.exp(sim_div) * logits_mask
        log_prob = sim_div - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-9)
        weighted_log_prob = tf.reduce_sum(weights * log_prob, axis=1)
        loss = -tf.reduce_mean(weighted_log_prob)
        return loss

class DynamicWeightedSupConLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.10, total_epochs=5, alpha_start=0, alpha_end=1.0, name='dynamic_weighted_supcon_loss'):
        super().__init__(name=name)
        self.temperature = temperature
        self.total_epochs = total_epochs
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.float32)

    def set_epoch(self, epoch):
        self.current_epoch.assign(epoch)

    def call(self, labels, features):
        features = tf.math.l2_normalize(features, axis=1)
        batch_size = tf.shape(features)[0]
        similarity = tf.matmul(features, features, transpose_b=True)
        sim_div = similarity / self.temperature

        labels = tf.reshape(labels, [-1, 1])
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
        logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
        mask *= logits_mask

        pos_mask = mask
        neg_mask = 1.0 - tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

        easy_weights = pos_mask / (tf.reduce_sum(pos_mask, axis=1, keepdims=True) + 1e-9)
        sim_jk_neg = similarity * neg_mask + (-1e6) * (1.0 - neg_mask)
        max_sim_jk_neg = tf.reduce_max(sim_jk_neg, axis=1)

        def compute_hard_weights(i):
            pos_idx = tf.where(pos_mask[i] > 0)[:, 0]
            w_i = tf.zeros([batch_size], dtype=tf.float32)
            def has_pos():
                pos_weights = tf.gather(max_sim_jk_neg, pos_idx)
                exp_weights = tf.exp(pos_weights)
                norm_weights = exp_weights / (tf.reduce_sum(exp_weights) + 1e-9)
                return tf.tensor_scatter_nd_update(w_i, tf.expand_dims(pos_idx, 1), norm_weights)
            return tf.cond(tf.shape(pos_idx)[0] > 0, has_pos, lambda: w_i)
        hard_weights = tf.map_fn(compute_hard_weights, tf.range(batch_size), dtype=tf.float32)

        progress = self.current_epoch / self.total_epochs
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        alpha = tf.clip_by_value(alpha, 0.0, 1.0)
        weights = (1.0 - alpha) * easy_weights + alpha * hard_weights

        exp_sim = tf.exp(sim_div) * logits_mask
        log_prob = sim_div - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-9)
        weighted_log_prob = tf.reduce_sum(weights * log_prob, axis=1)
        loss = -tf.reduce_mean(weighted_log_prob)
        return loss

# 损失实例化
if WEIGHT_TYPE == "dynamic":
    supcon_loss_fn = DynamicWeightedSupConLoss(temperature=0.1, total_epochs=epochs, alpha_start=0.0, alpha_end=1.0)
    alpha = 0.25
elif WEIGHT_TYPE == "static":
    supcon_loss_fn = SupConLoss(temperature=0.1)
    alpha = alpha_static

# ======= 6. 构建模型 =======
def build_bert_dpcnn_supcon_model(max_seq_length=128, feature_dim=256, num_blocks=2):
    bert_model = TFBertModel.from_pretrained(bert_dir)
    input_ids = keras.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_ids')
    attention_mask = keras.Input(shape=(max_seq_length,), dtype=tf.int32, name='attention_mask')
    token_type_ids = keras.Input(shape=(max_seq_length,), dtype=tf.int32, name='token_type_ids')

    bert_outputs = bert_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    seq_out = bert_outputs.last_hidden_state  # (batch, seq, hidden)

    def dpcnn_block(x, filters):
        branch2 = keras.layers.Conv1D(filters, 2, padding='same', activation='mish')(x)
        branch3 = keras.layers.Conv1D(filters, 3, padding='same', activation='mish')(x)
        branch4 = keras.layers.Conv1D(filters, 4, padding='same', activation='mish')(x)
        out = keras.layers.concatenate([branch2, branch3, branch4])
        out = keras.layers.BatchNormalization()(out)
        if out.shape[-1] != x.shape[-1]:
            x = keras.layers.Conv1D(out.shape[-1], 1, padding='same')(x)
        out = keras.layers.add([x, out])
        out = keras.layers.Activation('mish')(out)
        out = keras.layers.MaxPooling1D(3, strides=2, padding='same')(out)
        return out

    x = seq_out
    for _ in range(num_blocks):
        x = dpcnn_block(x, 250)
        x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dropout(0.3)(x)

    features = keras.layers.Dense(feature_dim, activation='relu')(x)
    features = keras.layers.BatchNormalization()(features)
    features = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="feature_norm")(features)
    cls_output = keras.layers.Dense(1, activation='sigmoid', name='classifier')(features)

    model = keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=[cls_output, features])
    return model

# ======= 7. 优化器设置（warmup+递减）=======
model = build_bert_dpcnn_supcon_model(max_length, feature_dim, num_blocks)

if y_train is not None:
    num_train_steps = int(np.ceil(len(y_train) / batch_size) * epochs)
    num_warmup_steps = int(num_train_steps * 0.1)
    optimizer, lr_schedule = create_optimizer(
        init_lr=2e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        weight_decay_rate=0.01
    )
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits, features = model(inputs, training=True)
        cls_loss = tf.reduce_mean(keras.losses.binary_crossentropy(labels, logits))
        supcon_loss = supcon_loss_fn(labels, features)
        loss = cls_loss + alpha * supcon_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, logits

def set_supcon_epoch(epoch):
    if hasattr(supcon_loss_fn, 'set_epoch'):
        supcon_loss_fn.set_epoch(epoch)

# ======= 8. 训练 =======
if y_train is not None:
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2048).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    for epoch in range(epochs):
        set_supcon_epoch(epoch)
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # ---------- Train ----------
        losses = []
        y_true_epoch, y_pred_epoch = [], []
        for batch, (inputs, labels) in enumerate(train_ds):
            loss, logits = train_step(inputs, tf.expand_dims(labels, -1))
            pred = (logits.numpy().reshape(-1) > 0.5).astype(np.int32)
            y_true_epoch.append(labels.numpy())
            y_pred_epoch.append(logits.numpy().reshape(-1))
            losses.append(loss.numpy())
        y_true_epoch = np.concatenate(y_true_epoch)
        y_pred_epoch = np.concatenate(y_pred_epoch)
        y_pred_label = (y_pred_epoch > 0.5).astype(int)
        acc = accuracy_score(y_true_epoch, y_pred_label)
        precision = precision_score(y_true_epoch, y_pred_label, zero_division=0)
        recall = recall_score(y_true_epoch, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_epoch, y_pred_label, zero_division=0)
        print(
            f"Train loss: {np.mean(losses):.4f}  acc: {acc:.4f}  precision: {precision:.4f}  recall: {recall:.4f}  f1: {f1:.4f}")

        # ---------- Validation ----------
        val_logits = []
        val_labels = []
        val_losses = []
        for val_inputs, val_y in val_ds:
            out = model(val_inputs, training=False)
            val_pred = out[0].numpy()
            val_logits.append(val_pred)
            val_labels.append(val_y.numpy())
            val_loss = tf.reduce_mean(keras.losses.binary_crossentropy(val_y, val_pred)).numpy()
            val_losses.append(val_loss)
        preds = np.concatenate(val_logits).reshape(-1)
        trues = np.concatenate(val_labels)
        preds_label = (preds > 0.5).astype(int)
        val_acc = accuracy_score(trues, preds_label)
        val_precision = precision_score(trues, preds_label, zero_division=0)
        val_recall = recall_score(trues, preds_label, zero_division=0)
        val_f1 = f1_score(trues, preds_label, zero_division=0)
        mean_val_loss = np.mean(val_losses)
        print(f"Validation: acc={val_acc:.4f}  precision={val_precision:.4f}  recall={val_recall:.4f}  f1={val_f1:.4f}  loss={mean_val_loss:.4f}")

    # ========== 9. 保存模型 ==========
    model.save_weights(save_model_weights)
    print(f"训练完成，权重保存为 {save_model_weights}")
else:
    print("未检测到训练集标签，不进行训练，仅推理。")

# ========== 10. 测试集推理 ==========
print("\n===== 测试集推理 =====")
model = build_bert_dpcnn_supcon_model(max_length, feature_dim, num_blocks)
model.load_weights(save_model_weights)
print("模型权重加载成功")

test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)
all_probs, all_features = [], []
for batch in test_dataset:
    logits, features = model(batch, training=False)
    all_probs.append(logits.numpy())
    all_features.append(features.numpy())
probs = np.concatenate(all_probs).reshape(-1)
features = np.concatenate(all_features)
preds = (probs > 0.5).astype(int)

# ========== 11. 保存测试结果 ==========
result_df = pd.DataFrame({
    'text': texts_test,
    'probability': probs,
    'prediction': preds
})
for i in range(features.shape[1]):
    result_df[f'feature_{i}'] = features[:, i]
output_csv = os.path.join(output_dir, 'test_results.csv')
result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"测试结果已保存至 {output_csv}")

# ========== 12. 如果有标签，输出评估 ==========
if y_test is not None:
    y_true = y_test
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    print("\n===== 评估结果 =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
else:
    print("\n测试集未包含标签，无法计算评估指标")

# ========== 13. 预测样例展示 ==========
print("\n===== 预测样例 =====")
print(result_df.head(10))
