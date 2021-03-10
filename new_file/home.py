import tensorflow as tf
y_true = [[0., 1.], [1., 1.], [1., 1.]]
y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
print(tf.expand_dims(y_true, 1).shape)
print(tf.expand_dims(y_pred, 0).shape)
loss = tf.keras.losses.cosine_similarity(tf.expand_dims(y_true, 1), tf.expand_dims(y_pred, 0))
print(loss.numpy())
l = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)
print(l.numpy())
# 今日baseline
# deepsurv cph deephit cfr
l = tf.keras.losses.cosine_similarity([[0., 1.], [0., 1.], [0., 1.]], y_pred, axis=1)
print(l.numpy())
