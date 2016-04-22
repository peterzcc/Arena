import os
import glob
import numpy

'''
preparing data for vot13, vot14 and vot15
note that duplicate video data will be removed
'''

vot_path = '/home/sliay/Documents/MDNet/dataset/VOT'
output_path = '/home/sliay/Documents/MDNet/dataset/VOT'
year_list = ['2013', '2014', '2015']

vot_list_file = open(os.path.join(output_path, 'vot.lst'), 'w')
vot_list = []
for year in year_list:
    video_list = []
    with open(os.path.join(vot_path, year, 'list.txt')) as fin:
        for line in fin:
            video_list.append(line.split()[0])

    for video in video_list:
        if any(video == s for s in vot_list):
            continue
        vot_list.append(video)
        img_list = sorted(glob.glob(os.path.join(vot_path, year, video, '*.jpg')))
        print video, year
        if year == '2013':
            roi = numpy.loadtxt(os.path.join(vot_path, year, video, 'groundtruth.txt'), delimiter=',')
        else:
            region = numpy.loadtxt(os.path.join(vot_path, year, video, 'groundtruth.txt'), delimiter=',')
            if region.shape[1] > 4:
                cx = region[:, 0::2].mean(axis=1)
                cy = region[:, 1::2].mean(axis=1)
                x1 = region[:, 0::2].min(axis=1)
                x2 = region[:, 0::2].max(axis=1)
                y1 = region[:, 1::2].min(axis=1)
                y2 = region[:, 1::2].max(axis=1)
                A1 = numpy.sqrt(numpy.sum((region[:, 0:2] - region[:, 2:4]) ** 2, axis=1)) * numpy.sqrt(numpy.sum((region[:, 2:4] - region[:, 4:6]) ** 2, axis=1))
                A2 = (x2 - x1) * (y2 - y1)
                s = numpy.sqrt(A1 / A2)
                w = s * (x2 - x1) + 1
                h = s * (y2 - y1) + 1
                roi = numpy.zeros((region.shape[0], 4))
                roi[:, 0] = x1
                roi[:, 1] = y1
                roi[:, 2] = w
                roi[:, 3] = h

        image_list_file = open(os.path.join(output_path, video+'.lst'), 'w')
        print roi.shape
        print len(img_list)
        assert roi.shape[0] > 0 and len(img_list) > 0
        for i in xrange(min(roi.shape[0], len(img_list))):
            one_line = '%s %d %d %d %d\n' % (img_list[i], roi[i, 0], roi[i, 1], roi[i, 2], roi[i, 3])
            image_list_file.write(one_line)
        image_list_file.close()
        one_line = '%s\n' % os.path.join(output_path, video+'.lst')
        vot_list_file.write(one_line)

vot_list_file.close()
