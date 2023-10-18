boolean_agegypti = True  # @param {type:"boolean"}
boolean_albopictus = False  # @param {type:"boolean"}
boolean_anopheles = True  # @param {type:"boolean"}
boolean_culex = False  # @param {type:"boolean"}
boolean_culiseta = True  # @param {type:"boolean"}
boolean_japonicus_koreicus = True  # @param {type:"boolean"}
boolean_values = [boolean_agegypti, boolean_albopictus,
                  boolean_anopheles, boolean_culex,
                  boolean_culiseta, boolean_japonicus_koreicus
                  ]
cut_and_mix_classes = tf.convert_to_tensor(boolean_values, dtype=tf.bool)


def cutmix(image, label, PROBABILITY=1.0, AUG_BATCH=BATCH_SIZE):
    # https://www.kaggle.com/code/cdeotte/cutmix-and-mixup-on-gpu-tpu?scriptVersionId=36764100&cellId=17
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with cutmix applied
    DIM = IMAGE_SIZE[0]
    CLASSES = 6

    imgs = []
    labs = []
    for j in range(AUG_BATCH):
        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
        # CHOOSE RANDOM IMAGE TO CUTMIX WITH
        k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
        if tf.math.logical_not(tf.math.logical_or(cut_and_mix_classes[tf.argmax(label[j])], cut_and_mix_classes[tf.argmax(label[k])])):
            P = tf.constant(0, dtype=tf.int32)
        # CHOOSE RANDOM LOCATION
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
        WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        # MAKE CUTMIX IMAGE
        one = image[j, ya:yb, 0:xa, :]
        two = image[k, ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
        imgs.append(img)
        # MAKE CUTMIX LABEL
        a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
        if len(label.shape) == 1:
            lab1 = tf.one_hot(label[j], CLASSES)
            lab2 = tf.one_hot(label[k], CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs), (AUG_BATCH, CLASSES))
    return image2, label2


def mixup(image, label, PROBABILITY=1.0, AUG_BATCH=BATCH_SIZE):
    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
    # output - a batch of images with mixup applied
    DIM = IMAGE_SIZE[0]
    CLASSES = 6

    imgs = []
    labs = []
    for j in range(AUG_BATCH):
        # DO MIXUP WITH PROBABILITY DEFINED ABOVE
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
        # CHOOSE RANDOM
        k = tf.cast(tf.random.uniform([], 0, AUG_BATCH), tf.int32)
        a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
        # MAKE MIXUP IMAGE
        img1 = image[j,]
        img2 = image[k,]
        if tf.math.logical_not(tf.math.logical_or(cut_and_mix_classes[tf.argmax(label[j])], cut_and_mix_classes[tf.argmax(label[k])])):
            a = tf.constant(0.0, dtype=tf.float32)
        imgs.append((1 - a) * img1 + a * img2)
        # MAKE CUTMIX LABEL
        if len(label.shape) == 1:
            lab1 = tf.one_hot(label[j], CLASSES)
            lab2 = tf.one_hot(label[k], CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image2 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
    label2 = tf.reshape(tf.stack(labs), (AUG_BATCH, CLASSES))
    return image2, label2


def transform(image, label, AUG_BATCH=BATCH_SIZE):
    # THIS FUNCTION APPLIES BOTH CUTMIX AND MIXUP
    DIM = IMAGE_SIZE[0]
    CLASSES = 6
    SWITCH = 0.5
    CUTMIX_PROB = 1.0
    MIXUP_PROB = 1.0
    # FOR SWITCH PERCENT OF TIME WE DO CUTMIX AND (1-SWITCH) WE DO MIXUP
    image2, label2 = cutmix(image, label, CUTMIX_PROB)
    image3, label3 = mixup(image, label, MIXUP_PROB)
    imgs = []
    labs = []
    for j in range(AUG_BATCH):
        P = tf.cast(tf.random.uniform([], 0, 1) <= SWITCH, tf.float32)
        imgs.append(P * image2[j,] + (1 - P) * image3[j,])
        labs.append(P * label2[j,] + (1 - P) * label3[j,])
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image4 = tf.reshape(tf.stack(imgs), (AUG_BATCH, DIM, DIM, 3))
    label4 = tf.reshape(tf.stack(labs), (AUG_BATCH, CLASSES))
    return image4, label4


if not DISABLE_CUTMIX:
    training_dataset_augmented = training_dataset_augmented.unbatch()
    training_dataset_augmented = training_dataset_augmented.repeat().batch(BATCH_SIZE).map(transform)

    row = 6
    col = 4
    row = min(row, BATCH_SIZE // col)
    for (img, label) in training_dataset_augmented:
        plt.figure(figsize=(15, int(15 * row / col)))
        for j in range(row * col):
            plt.subplot(row, col, j + 1)
            plt.axis('off')
            plt.imshow(img[j,])
        plt.show()
        break
else:
    training_dataset_augmented = training_dataset_augmented.repeat()




def transform_gridmask(image, inv_mat, image_shape):
    h, w, c = image_shape
    cx, cy = w//2, h//2
    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)
    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])
    new_zs = tf.ones([h*w], dtype=tf.int32)
    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))
    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)
    clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)
    clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)
    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)
    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))
    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))
    new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))
    new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))
    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)
    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)
    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))
    rotated_image_channel = list()
    for i in range(c):
        vals = rotated_image_values[:,i]
        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])
        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))
    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])

def random_rotate(image, angle, image_shape):
    def get_rotation_mat_inv(angle):
        # transform to radian
        angle = math.pi * angle / 180
        cos_val = tf.math.cos(angle)
        sin_val = tf.math.sin(angle)
        one = tf.constant([1], tf.float32)
        zero = tf.constant([0], tf.float32)
        rot_mat_inv = tf.concat([cos_val, sin_val, zero, -sin_val, cos_val, zero, zero, zero, one], axis=0)
        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])
        return rot_mat_inv
    angle = float(angle) * tf.random.normal([1],dtype='float32')
    rot_mat_inv = get_rotation_mat_inv(angle)
    return transform_gridmask(image, rot_mat_inv, image_shape)


def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):
    h, w = image_height, image_width
    hh = int(np.ceil(np.sqrt(h*h+w*w)))
    hh = hh+1 if hh%2==1 else hh
    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)
    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)

    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)
    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)
    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)

    for i in range(0, hh//d+1):
        s1 = i * d + st_h
        s2 = i * d + st_w
        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)
        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)

    x_clip_mask = tf.logical_or(x_ranges < 0 , x_ranges > hh-1)
    y_clip_mask = tf.logical_or(y_ranges < 0 , y_ranges > hh-1)
    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)

    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))
    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))

    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])
    x_ranges = tf.repeat(x_ranges, hh)
    y_ranges = tf.repeat(y_ranges, hh)

    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))
    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))

    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])
    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)

    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])
    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)

    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)

    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])
    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)

    return mask

def apply_grid_mask(image, image_shape):
    AugParams = {
        'd1' : 100,
        'd2': 160,
        'rotate' : 45,
        'ratio' : 0.3
    }
    mask = GridMask(image_shape[0], image_shape[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'], AugParams['ratio'])
    if image_shape[-1] == 3:
        mask = tf.concat([mask, mask, mask], axis=-1)
    return image * tf.cast(mask,tf.float32)

def gridmask(img_batch, label_batch):
    return apply_grid_mask(img_batch, (IMAGE_SIZE[0],IMAGE_SIZE[1] 3)), label_batch