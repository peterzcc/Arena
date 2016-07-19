import os
import glob
import numpy
import re


'''
prepare training data for vot14 and vot15 using sequences from {alov, otb100}-{vot14,15}
'''
output_path = 'D://HKUST//2-2//learning-to-track//datasets//training_for_vot'
otb_path = 'D://HKUST//2-2//learning-to-track//datasets//OTB100'
alov_path = 'D://HKUST//2-2//learning-to-track//datasets//alov300/'

folder_path = os.path.dirname(os.path.abspath(__file__))

list_file_14 = open(os.path.join(output_path, 'training_vot14.lst'), 'w')
list_file_15 = open(os.path.join(output_path, 'training_vot15.lst'), 'w')
training_list = []


# process otb data
with open(os.path.join(folder_path, 'otb-vot14.txt')) as fin:
    for line in fin:
        name = line.split()[0]
        name = name.split('-')
        if len(name) == 2:
            roi_str = 'groundtruth_rect.'+name[1]+'.txt'
        elif name[0] == 'Human4':
            roi_str = 'groundtruth_rect.2.txt'
        else:
            roi_str = 'groundtruth_rect.txt'
        name = name[0]
        roi = []
        img_files = sorted(glob.glob(os.path.join(otb_path, name, 'img', '*')))
        with open(os.path.join(otb_path, name, roi_str)) as f:
            for fline in f:
                tmp = re.split('[ ,;\t]', fline)
                roi.append([float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3])])
        roi = numpy.array(roi)
        start_index = 0
        if name == 'David':
            start_index = 299
        image_list_file = open(os.path.join(output_path, name+'.lst'), 'w')
        for i in range(roi.shape[0]):
            one_line = '%s %d %d %d %d\n' % (img_files[start_index+i], roi[i,0], roi[i,1], roi[i,2], roi[i,3])
            image_list_file.write(one_line)
        image_list_file.close()
        one_line = '%s\n' % os.path.join(output_path, name+'.lst')
        list_file_14.write(one_line)

with open(os.path.join(folder_path, 'otb-vot15.txt')) as fin:
    for line in fin:
        name = line.split()[0]
        name = name.split('-')
        if len(name) == 2:
            roi_str = 'groundtruth_rect.'+name[1]+'.txt'
        elif name[0] == 'Human4':
            roi_str = 'groundtruth_rect.2.txt'
        else:
            roi_str = 'groundtruth_rect.txt'
        name = name[0]
        img_files = sorted(glob.glob(os.path.join(otb_path, name, 'img', '*')))
        roi = []
        with open(os.path.join(otb_path, name, roi_str)) as f:
            for fline in f:
                tmp = re.split('[ ,;\t]', fline)
                roi.append([float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3])])
        roi = numpy.array(roi)
        start_index = 0
        if name == 'David':
            start_index = 299
        image_list_file = open(os.path.join(output_path, name+'.lst'), 'w')
        for i in range(roi.shape[0]):
            one_line = '%s %d %d %d %d\n' % (img_files[start_index+i], roi[i,0], roi[i,1], roi[i,2], roi[i,3])
            image_list_file.write(one_line)
        image_list_file.close()
        one_line = '%s\n' % os.path.join(output_path, name+'.lst')
        list_file_15.write(one_line)


# process alov data
video_path = os.path.join(alov_path, 'imagedata++')
roi_path = os.path.join(alov_path, 'alov300++_rectangleAnnotation_full')
categories = sorted(os.listdir(video_path))
alov_omit_list = [
    '01-Light_video00016',
    '01-Light_video00022',
    '01-Light_video00023',
    '02-SurfaceCover_video00012',
    '03-Specularity_video00003',
    '03-Specularity_video00012',
    '04-Transparency_video00009',
    '05-Shape_video00003',
    '05-Shape_video00013',
    '06-MotionSmoothness_video00004',
    '06-MotionSmoothness_video00022',
    '09-Confusion_video00014',
    '11-Occlusion_video00008',
    '11-Occlusion_video00012',
    '10-LowContrast_video00013',
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
        cx = (roi[:, 0] + roi[:, 2]) / 2
        cy = (roi[:, 1] + roi[:, 3]) / 2
        wh = abs(roi[:, 2:4] - roi[:, 0:2]) + 1
        roi[:, 0] = cx - wh[:, 0]/2
        roi[:, 1] = cy - wh[:, 1]/2
        roi[:, 2] = wh[:, 0]
        roi[:, 3] = wh[:, 1]
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
        list_file_14.write(line)
        list_file_15.write(line)

list_file_14.close()
list_file_15.close()

