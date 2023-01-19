
    
def _save_debug_data(self, data_info:list, iter :int, results_pred , folder_name: str , input_folder : str, ann_folder : str ) :
    import os , shutil , cv2
    #input path = data_info.img_prefix + '\' + data_info.img_info.filename
    #dst path = log folder + iter index + label 
    input_filename = input_folder + '/' + data_info['filename']
    ann_filename = ann_folder + '/' + data_info['ann']['seg_map']
    savefolder =  self.out_dir + folder_name 
    fileName = os.path.splitext(data_info['filename'])
    savefilepath = savefolder + '/' + fileName[0] + '_score_'+ str(round(results_pred,3)) + '_input' + fileName[1]
    updatedfolder = os.path.dirname(savefilepath)
    if(os.path.isdir(updatedfolder) == False):
        os.makedirs(updatedfolder) #mkdir
    shutil.copyfile(input_filename,savefilepath)
    
    savefilepath = savefolder + '/' + fileName[0] + '_score_'+ str(round(results_pred,3)) + '_groundtruth' + fileName[1]
    ann_image = cv2.imread(ann_filename) 
    ann_image = ann_image * 100 
    cv2.imwrite(savefilepath, ann_image)
    # shutil.copyfile(ann_filename,savefilepath)

def function():
    t_idx = 0
    result_list = []
    if hasattr(self.dataloader.dataset, 'img_infos'):
        for data_info in self.dataloader.dataset.img_infos:
            gt_image = data_info['ann']['seg_map'] #file_name
            result_dict = dict(input_name = data_info['filename'], ann_name = gt_image, key_score = key_score)
            # gt_label2 = gt_label[0] # base 0
            if key_score < 0.9 :
                folder_name = '/iter_' + str(runner.iter) + '_false' 
                self._save_debug_data(data_info, runner.iter, key_score, folder_name, self.dataloader.dataset.img_dir, self.dataloader.dataset.ann_dir)
            else :
                folder_name = '/iter_' + str(runner.iter) + '_true'
                self._save_debug_data(data_info, runner.iter, key_score, folder_name, self.dataloader.dataset.img_dir, self.dataloader.dataset.ann_dir)
            result_list.append(result_dict)
            t_idx = t_idx+1

    #save file 
    import csv 
    with open(self.out_dir + '/iter_' + str(runner.iter) + '_pred.csv','a',newline='') as f :
            writer = csv.writer(f)
            writer.writerow(result_list[0].keys())
            for item in result_list:
                writer.writerow(item.values())


import os
from glob import glob
import cv2


abs_target_path = 'Z:/AnomalyClassification/open-mmlab/mmsegmentation-kmlee/work_dirs/unet-iter-400k/iter_399999_true'
palette = [[10, 10, 10], [50, 50, 220], [255, 100, 100]]


def calc_image(strLabel = "anomal"):

    pred_color = (255, 100, 100)
    gt_color = 200
    if strLabel == "normal" :
        pred_color = (50, 50, 220)
        gt_color = 100

    # load images 
    imagelist = list(map(lambda path: path, glob(rf"{abs_target_path}\*_groundtruth.bmp")))
    predlist = list(map(lambda path: path , glob(rf"{abs_target_path}\*_color_seg.bmp")))
    
    idx = 0
    result_list = []
    for image in imagelist :
        gt_path = image 
        pred_path = predlist[idx]
        # load image 
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_path)
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        # get anomaly image 
        gt_thres = cv2.inRange(gt_image,gt_color-1, gt_color+1)
        pred_thres = cv2.inRange(pred_image,pred_color,pred_color)
        # count blob 
        n, labels, stats, _ = cv2.connectedComponentsWithStats(pred_thres)
        gt_n, gt_labels, gt_stat, _ =  cv2.connectedComponentsWithStats(gt_thres)

        # default : image_name, gt_n, gt_area, pred_n, pred_area_list, isoverlap, overlap_rate 
        gt_dir, gt_file_name = os.path.split(gt_path)
        if gt_n >= 2 :
            (gt_x, gt_y, gt_w, gt_h, gt_area) = gt_stat[1]
        else:
            (gt_x, gt_y, gt_w, gt_h, gt_area) = (0,0,0,0,0)

        # save info 
        area_list = []
        for i in range(1, n):
            (x, y, w, h, pred_area) = stats[i]
            area_list.append(pred_area)

        #overlap info 
        overlap = gt_thres & pred_thres
        overlap_n, overlap_labels, overlap_stat, _ = cv2.connectedComponentsWithStats(overlap)
            
        isOverlap = overlap_n >= 2
        if(isOverlap) :
            (overlap_x, overlap_y, overlap_w, overlap_h, overlap_area) = overlap_stat[1]
            result_dict = dict(filename = gt_file_name, gt_n = gt_n-1, gt_area = gt_area, pred_n = n-1, pred_area = area_list, isoverlap = isOverlap, overlap_stat =overlap_area, overlap_rate = overlap_area / gt_area )
            overlap_path = gt_path.replace('_groundtruth','_overlap_'+ strLabel +'_'+str(i))
            cv2.imwrite(overlap_path,(overlap*255))
            result_list.append(result_dict)
        else :
            result_dict = dict(filename = gt_file_name, gt_n = gt_n-1, gt_area = gt_area, pred_n = n-1, pred_area = area_list, isoverlap = False, overlap_stat = 0, overlap_rate = 0 )
            result_list.append(result_dict)

        #file index
        idx = idx+1

    #save file 
    import csv 
    with open(abs_target_path + '/result_pred_'+strLabel+'_image_v2.csv','a',newline='') as f :
            writer = csv.writer(f)
            writer.writerow(result_list[0].keys())
            for item in result_list:
                writer.writerow(item.values())

def calc_blob(strLabel = "anomal"):

    pred_color = (255, 100, 100)
    gt_color = 200
    if strLabel == "normal" :
        pred_color = (50, 50, 220)
        gt_color = 100

    # load images 
    imagelist = list(map(lambda path: path, glob(rf"{abs_target_path}\*_groundtruth.bmp")))
    predlist = list(map(lambda path: path , glob(rf"{abs_target_path}\*_color_seg.bmp")))
    
    idx = 0
    result_list = []
    for image in imagelist :
        gt_path = image 
        pred_path = predlist[idx]
        gt_dir, gt_file_name = os.path.split(gt_path)
        # load image 
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(pred_path)
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
        # get binary image 
        gt_thres = cv2.inRange(gt_image,gt_color-1, gt_color+1)
        pred_thres = cv2.inRange(pred_image,pred_color,pred_color)
        # count blob 
        total_blob, labels, stats, _ = cv2.connectedComponentsWithStats(pred_thres)
        gt_n, gt_labels, gt_stat, _ =  cv2.connectedComponentsWithStats(gt_thres)

        if gt_n >= 2 :
            (gt_x, gt_y, gt_w, gt_h, gt_area) = gt_stat[1]
        else:
            (gt_x, gt_y, gt_w, gt_h, gt_area) = (0,0,0,0,0)

        # overlap label number 
        overlap_label_list = []
        overlap_pred_area_list = [] # overlap_size 
        overlap_area_list = [] # pred_size
        for i in range(1, total_blob):          
            #overlap info 
            overlap = ( labels == i ) 
            overlap = gt_thres & overlap
            overlap_n, overlap_labels, overlap_stat, _ = cv2.connectedComponentsWithStats(overlap)
            
            isOverlap = overlap_n >= 2
            if(isOverlap) :
                overlap_label_list.append(i)
                (x, y, w, h, pred_area) = stats[i]
                overlap_pred_area_list.append(pred_area)
                (x, y, w, h, overlap_area) = overlap_stat[1]
                overlap_area_list.append(overlap_area)

        #overlap info 
        if(len(overlap_label_list)>=1):
            overlap = gt_thres & pred_thres
            sum_value = sum(overlap_area_list)

            result_dict = dict(filename = gt_file_name, gt_n = gt_n-1, gt_area = gt_area, pred_n = total_blob-1, pred_anomal_idx = overlap_label_list, pred_area = overlap_pred_area_list, isoverlap = True, overlap_stat = sum_value, overlap_rate = sum_value / gt_area )
            overlap_path = gt_path.replace('_groundtruth','_overlap_'+ strLabel +'_'+ '['+','.join(map(str,overlap_label_list))+']')
            cv2.imwrite(overlap_path,(overlap))
            result_list.append(result_dict)
        else: # 겹침이 없음. False Negative 
            result_dict = dict(filename = gt_file_name, gt_n = gt_n-1, gt_area = gt_area, pred_n =  total_blob-1, pred_anomal_idx = [-1], pred_area = [-1], isoverlap = False, overlap_stat = 0, overlap_rate = 0 )
            result_list.append(result_dict)

            import shutil
            if(gt_n-1>=1): # save the false negative 
                analy_path = os.path.join(gt_dir, 'analy_fn')
                if(os.path.isdir(analy_path) == False):
                    os.makedirs(analy_path) #mkdir
                #input
                in_filename = gt_file_name.replace('_groundtruth','_input')
                out_path = analy_path + '/' + in_filename
                shutil.copyfile(gt_dir+'/'+in_filename, out_path)
                #gt
                out_path = analy_path + '/' + gt_file_name
                shutil.copyfile(gt_path, out_path)
                #result
                pred_dir, pred_file_name = os.path.split(pred_path)
                out_path = analy_path + '/' + pred_file_name
                shutil.copyfile(pred_path, out_path)

        # 겹침영역 없음 분석
        false_call_list = [*range(1,total_blob)] #range to list 
        for i in overlap_label_list :
            false_call_list.remove(i)

        for i in false_call_list:
            (x, y, w, h, pred_area) = stats[i]
            result_dict = dict(filename = gt_file_name, gt_n = gt_n-1, gt_area = gt_area, pred_n = total_blob-1, pred_anomal_idx = [i], pred_area = [pred_area], isoverlap = False, overlap_stat = 0, overlap_rate = 0 )
            result_list.append(result_dict)
            
            import shutil
            analy_path = os.path.join(gt_dir, 'analy_fp')
            if(os.path.isdir(analy_path) == False):
                os.makedirs(analy_path) #mkdir
            #input
            in_filename = gt_file_name.replace('_groundtruth','_input')
            out_path = analy_path + '/' + in_filename
            shutil.copyfile(gt_dir+'/'+in_filename, out_path)
            #gt
            out_path = analy_path + '/' + gt_file_name
            shutil.copyfile(gt_path, out_path)
            #result
            pred_dir, pred_file_name = os.path.split(pred_path)
            pred_file_name = pred_file_name.replace('_color_seg','_color_seg_label_'+str(i))
            out_path = analy_path + '/' + pred_file_name
            overlap = ( labels == i ) 
            cv2.imwrite(out_path,overlap*255)


        if total_blob == 1 : # 아무것도 검출되지 않음, background 만 있음  
            result_dict = dict(filename = gt_file_name, gt_n = gt_n-1, gt_area = gt_area, pred_n = total_blob-1, pred_anomal_idx = [0], pred_area = [0], isoverlap = False, overlap_stat = 0, overlap_rate = 0 )
            result_list.append(result_dict)
        #file index
        idx = idx+1

    #save file 
    import csv 
    with open(abs_target_path + '/result_pred_'+ strLabel +'_blob_v3.csv','a',newline='') as f :
            writer = csv.writer(f)
            writer.writerow(result_list[0].keys())
            for item in result_list:
                writer.writerow(item.values())

if __name__ == "__main__":
    #calc_blob("normal") # nomal 
    #calc_image("normal")
    calc_blob("anormal") # anomal 
    #calc_image("anormal")
