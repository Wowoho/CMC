
def Cal_CMC(net:capNetRID.CapNetForRid, num_id):
    def get_feature(image_1, image_2):
        f1, f2 = net.feed_forward(image_1, image_2)
        return f1, f2

    def get_dist(f1, f2):
        return np.sum(np.square(f1 - f2))

    def get_images(i):
        while True:
            x, y = random.randint(0, 9), random.randint(0, 9)
            while x == y:
                y = random.randint(0, 9)

            path_x = '%s/detected/%s/%04d_%02d.jpg' % (path, data_set, i, x)
            path_y = '%s/detected/%s/%04d_%02d.jpg' % (path, data_set, i, y)
            # print(i, os.path.exists(path_x), os.path.exists(path_y))
            if os.path.exists(path_x) and os.path.exists(path_y):
                break

        img_x = cv2.imread(path_x)
        img_y = cv2.imread(path_y)
        return img_x, img_y

    feature_1 = []
    feature_2 = []
    cmc = np.array([0 for i in range(num_id)])
    for i in range(num_id):
        img_x, img_y = get_images(i)

        img_x = np.array(cv2.resize(img_x, (image_width, image_height))).reshape([1, image_height, image_width, 3])
        img_y = np.array(cv2.resize(img_y, (image_width, image_height))).reshape([1, image_height, image_width, 3])

        fx, fy = get_feature(img_x, img_y)

        feature_1.append(fx)
        feature_2.append(fy)

    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)

    for i in range(num_id):
        rank_i = []
        for j in range(num_id):
            rank_i.append((get_dist(feature_1[i], feature_2[j]), j))

        rank_i.sort()

        for j in range(num_id):
            if rank_i[j][1] == i:
                cmc[j] += 1
                break

    for i in range(num_id-1):
        cmc[i+1] += cmc[i]

    print(cmc[:100])
