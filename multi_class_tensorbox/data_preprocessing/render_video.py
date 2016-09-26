import cv2


def video_render(path, s_idx, e_idx, video_path):

    for i in xrange(s_idx, e_idx):
        filename = path+'%d.png'%i
        print 'processing %s'%filename
        img = cv2.imread(filename)
        height, width, layers = img.shape

        video = cv2.VideoWriter(video_path, -1, 1, (width, height))
        video.write(img)
    cv2.destroyAllWindows()
    video.release()

video_render('/home/chaolan/test_results/detector_out_conf_0.6/', 1, 3000, '/home/chaolan/test_results/detector_out_conf_0.6/video.avi')