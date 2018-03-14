'''

Script to generate all maps plots

List of plots generated

1. Z stamping
    1.1. Z stage (from kymograph)
    1.2. Z stamp (curvefit)
    1.3. Z stamp optimal
    1.4. z Residues
2. Canonical Heartbeat Length
    2.1. Correlation curve
    2.2. Spectrum of correlation
3. Canonical heartbeat
    3.1. Diff matrix
    3.2. Diff matrix row
    3.3. Mean2Std curve
4. Phase stamping
    4.1. Phase stamps
    4.2. Phase stamp matchvals
5. Phi Z matrices

6. Canonical heartbeat correlation
    6.1.
'''

from maps import settings
from maps.settings import setting, read_setting_from_json, \
    display_current_settings, JOBS_DIR, display_step
from maps.helpers.misc import pickle_object, unpickle_object, make_or_clear_directory, write_log_data

job_list = next(os.walk(settings.JOBS_DIR))[1]

job_choice = 0

try:
    job_choice = int(sys.argv[1])
except:
    print '\n'.join(
        map(
            lambda x, y: '%d-%s' % (x, y),
            range(1, len(job_list) + 1),
            job_list
        )
    )

    while job_choice not in range(1, len(job_list) + 1):
        job_choice = int(raw_input('Select job:'))
curr_job = job_list[job_choice - 1]
read_setting_from_json(curr_job, save_bkups=SAVE_WORKSPACE)

try:
    run_name = sys.argv[2]
except:
    run_name = time.strftime('%d%b%y_%H%M')

try:
    execfile(os.path.join(JOBS_DIR, curr_job, 'local_settings.py'))
except:
    print '''Local setting override python file not found.
    Please create a local_settings.py file in the job directory.
    Using default config parameters'''
    Y_STAGE_CORRECTION = True
    WRITE_IMAGES_TO_DISK = True
    USE_PICKLED_ZSTAMPS = False
    USE_PICKLED_ZSTAMPS_STAT = False
    USE_PICKLED_PHASESTAMPS = False
    USE_PICKLED_DWT = False
    USE_PICKLED_CMHB = False
    USE_PICKLED_SAD = False
    USE_PICKLED_CANON_DWT = False
    USE_PICKLED_PHI_Z_MATRIX = False
    WRITE_DWT = True
    WRITE_FINAL_IMAGES = True


with open(os.path.join(settings.JOBS_DIR, curr_job, 'job_metadata.txt'), 'a') as metafile:
    data = '''
--------------------------------------------------------------------------------
Run name: {run_name}
Y correction: {ycorr}
Difference matrix method: {diff_method}
Multiprocessing: {mp}
'''.format(
        run_name=run_name,
        ycorr=setting['y_correction'],
        mp=settings.MULTIPROCESSING,
        diff_method=DIFF_MAT_METHOD
    )
    metafile.write(data)
