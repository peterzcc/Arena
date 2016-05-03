import os
import glob
import numpy


'''
prepare training data for OTB100 using sequences from {alov, vot13,14,15}-{otb100}
'''
output_path = 'D://HKUST//2-2//learning-to-track//datasets//training_for_otb100//'
otb_path = 'D://HKUST//2-2//learning-to-track//datasets//OTB100/'
vot_path = 'D://HKUST//2-2//learning-to-track//datasets//VOT'
alov_path = 'D://HKUST//2-2//learning-to-track//datasets//alov300/'

folder_path = os.path.dirname(os.path.abspath(__file__))
year_list = ['2013', '2014', '2015']

list_file = open(os.path.join(output_path, 'training_otb.lst'), 'w')
training_list = []

# store all the sequences from OTB100
otb_name = []
with open(os.path.join(folder_path, 'otb100.lst')) as fin:
    for line in fin:
        otb_name.append(line.split()[0])
otb_name.pop(otb_name.index('crossing'))
omit_list = ['car', 'bolt1', 'bolt2', 'face', 'gymnastics', 'hand2', 'motocross', 'motocross1', 'pedestrian1', 'pedestrian2', 'singer', 'skating', 'soccer1', 'tiger', 'iceskater2']

# process vot data
for year in year_list:
    video_list = []
    with open(os.path.join(vot_path, year, 'list.txt')) as fin:
        for line in fin:
            video_list.append(line.split()[0])

    for video in video_list:
        if any(video == s for s in training_list) or any(video == s for s in otb_name) or any(video == s for s in omit_list):
            continue
        training_list.append(video)
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
        list_file.write(one_line)


# process alov data
video_path = os.path.join(alov_path, 'imagedata++')
roi_path = os.path.join(alov_path, 'alov300++_rectangleAnnotation_full')
categories = sorted(os.listdir(video_path))
alov_omit_list = [
    '01-Light_video00007',
    '01-Light_video00015',
    '01-Light_video00016',
    '01-Light_video00021',
    '02-SurfaceCover_video00012',
    '06-MotionSmoothness_video00016',
    '06-MotionSmoothness_video00019',
    '08-Clutter_video00011',
    '09-Confusion_video00001',
    '11-Occlusion_video00010',
    '11-Occlusion_video00024',
    '11-Occlusion_video00025',
]

for cat in categories:
    video_per_category = sorted(os.listdir(os.path.join(video_path, cat)))
    gt_per_category = sorted(os.listdir(os.path.join(roi_path, cat)))
    counter = -1
    for gt in gt_per_category:
        counter += 1
        if any(video_per_category[counter] == s for s in alov_omit_list):
            continue
        roi = numpy.loadtxt(os.path.join(roi_path, cat, gt))
        index = numpy.int64(roi[:, 0])
        roi = roi[:, [3, 2, 1, 6]]
        roi[:, 2:4] = roi[:, 2:4] - roi[:, 0:2] + 1
        interp_roi = numpy.zeros((index[-1] - index[0] + 1, 4))
        # do linear interpolation
        for i in xrange(4):
            interp_roi[:, i] = numpy.interp(numpy.arange(index[0], index[-1]+1), index, roi[:, i])
        name = gt.split('.')[0]
        image_list_file = open(os.path.join(output_path, name+'.lst'), 'w')
        img_list = sorted(os.listdir(os.path.join(video_path, cat, video_per_category[counter])))
        for i in xrange(interp_roi.shape[0]):
            line = '%s %d %d %d %d\n' % (os.path.join(video_path, cat ,video_per_category[counter], img_list[i+index[0]-1]), interp_roi[i, 0], interp_roi[i, 1], interp_roi[i, 2], interp_roi[i, 3])
            image_list_file.write(line)
        image_list_file.close()
        line = '%s\n' % os.path.join(output_path, name+'.lst')
        list_file.write(line)

