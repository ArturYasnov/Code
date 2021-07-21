...

def collect_image(id):
    appnd_len = 10 - len(id)
    appnd = '0' * appnd_len
    id = appnd + id

    # collect image
    url = 'https://../{}.jpg?rule=gallery'.format(id[:2], id)
    image = requests.get(url).content

    if len(image) > 0:
        print('len>0')
        nparr = np.fromstring(image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        #cv2.imwrite('/Users/arthur/PycharmProjects/RepairQuality/img.jpg', img_np)
        return img_np
    else:
        print('none')
        return 0


...

