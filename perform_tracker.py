import os
import cv2
from components.main_tracker import TargetTracker
from datasets.read_seq import frame_loader
from functions import visualization as vis
import logging
from functions.log import logger
from functions.timer import Timer


def save_results(filename, results, data_type):

    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('write results to {}'.format(filename))

def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)

def eval_tracker(dataloader, data_type, result_filename, save_dir=None, show_image=True):
    if save_dir is not None:
        mkdirs(save_dir)

    tracker = TargetTracker()
    timer = Timer()
    results = []
    wait_time = 1
    for frame_id, batch in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1./max(1e-5, timer.average_time)))

        frame, det_tlwhs, det_scores, _, _ = batch

        # run tracking
        timer.tic()
        online_targets = tracker.update(frame, det_tlwhs, None)
        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            online_tlwhs.append(t.tlwh)
            online_ids.append(t.track_id)
        timer.toc()

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        online_res = vis.draw_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                      fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_res', online_res)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_res)

        key = cv2.waitKey(wait_time)
        key = chr(key % 128).lower()
        if key == 'q':
            exit(0)
        elif key == 'p':
            cv2.waitKey(0)
        elif key == 'a':
            wait_time = int(not wait_time)

    # save results
    save_results(result_filename, results, data_type)


def main(data_root='/media/bozorgta/My Passport/MOT16/train', det_root=None,
         seqs=('MOT16-09',), exp_name='BB', save_image=True, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdirs(result_root)
    data_type = 'mot'

    # perform tracking
    for seq in seqs:
        output_dir = os.path.join(data_root, 'outputs', seq) if save_image else None

        logger.info('start seq: {}'.format(seq))
        loader = frame_loader(data_root, det_root, seq)
        eval_tracker(loader, data_type, os.path.join(result_root, '{}.txt'.format(seq)),
                 save_dir=output_dir, show_image=show_image)


if __name__ == '__main__':
    # Choose sequences
     seqs_str = '''MOT16-05
                 MOT16-09'''
     seqs = [seq.strip() for seq in seqs_str.split()]

     main(data_root='/media/bozorgta/My Passport/MOT16/train',
          seqs=seqs,
          exp_name='mot16',
          show_image=True)
