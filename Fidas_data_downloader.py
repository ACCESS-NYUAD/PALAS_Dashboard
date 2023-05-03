#
# (C) 2023 ACCESS - Francesco Paparella. 
#
#----------------------------------------------------------------------
#
# Downloads the monthly data file from the Palas Fidas 200 S station
# installed by ACCESS. Save the data both in the native .txt tabular format
# and in the netcdf format.
# The script polls the Fidas every 10 minutes, downloading the last saved
# .txt data file. The datafile is then also converted in netCDF format.
#
#
from ftplib import FTP
from fnmatch import fnmatch
import time as systime
import os
import signal
import functools
import pandas
from netCDF4 import Dataset
import numpy as np

# IP number and credentials of the Fidas 200S FTP server and relevant paths
from Fidas_credentials import *

#------------------------------------------------
DownloadInterval = 60*10 # in seconds
MaxTimeAllowed = 60*3 # in seconds: ftp will be closed if transfer lasts longer
# Time in the netCDF file is in seconds elapsed from the basetime
# The time zone is saved in the time units info of the netcdf file.
basetime = np.datetime64('2020-01-01T00:00:00')
timezone = '+04:00'

#---------------------------------------------------------------------
# Define a timeout decorator to avoid hanging file transfers.
# If the timer expires an error is raised and the decorated function
# is interrupted.
# See: https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
#

class FTPTimeoutError(Exception):
    pass

def timeout(seconds=10, error_message='Timer expired'):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise FTPTimeoutError(error_message)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

#---------------------------------------------------------------------
def open_ftp_get_file_list(ipnum, username, password, folder):
    ftplink = FTP(ipnum)
    ftplink.login(user=username, passwd=password)
    ftplink.cwd(folder)
    foldercontent = []
    ftplink.retrlines('LIST', foldercontent.append)
    return ftplink, foldercontent

#---------------------------------------------------------------------
def files_to_be_downloaded(foldercontent, local_path):
    """Returns a list of files to be downloaded, chosen according to the
following criterion:
- add to the list the most recent data file.
- if that is not present in the local_path, add also the previous data file.
- continue adding until there are no data files or one present in local_path is found.
"""
    filenames = [line.split(' ')[-1] for line in foldercontent]
    filenames = [fn for fn in filenames if fnmatch(fn, 'DUSTMONITOR*.txt')]
    if filenames==[]:
        return None
    year_month = [(fn[-11:-7], fn[-6:-4]) for fn in filenames]
    year_month.sort()
    to_be_downloaded = []
    for year, month in reversed(year_month):
        for fn in filenames:
            if fnmatch(fn, f'DUSTMONITOR*{year}_{month}.txt'):
                fname = fn
                break
        to_be_downloaded.append(fname)  
        if os.path.exists(f"{local_path}/{fname}"):
            break
    return to_be_downloaded
    
#---------------------------------------------------------------------
@timeout(seconds=MaxTimeAllowed,
         error_message=f"Timeout while downloading data file.")
def _get_ftp_file(ftpobj, remote_fname, local_fname):
    with open(local_fname, 'wb') as fp:
        ftpobj.retrbinary(f"RETR {remote_fname}", fp.write)

def download_data_file(ftpobj, local_path, filename):
    try:
        _get_ftp_file(ftpobj=ftpobj,
                      remote_fname=filename,
                      local_fname=f"{local_path}/temporary_dust_file.txt")
        os.rename(f"{local_path}/temporary_dust_file.txt",
                  f"{local_path}/{filename}")
    except FTPTimeoutError as err:
        #ftpobj.quit() #this triggers an ftp error
        raise err

#-----------------------------------------------------------------
# Convert the data to netCDF format. For documentation on the units
# see the spreadsheet sent by <mara.brand@palas.de> on 3/28/23, 13:49
#
def convert_to_netCFD4(local_path, filename):
    dustdata = pandas.read_table(f"{local_path}/{filename}",
                                 parse_dates=[['date', 'time']])
    sizes_labels = dustdata.columns[43:121]
    #
    ncfile = Dataset(f"{local_path}/temporary_dust_file.nc",mode='w')
    ncfile.title=f"PALAS Fidas 200 S data - downloaded on {systime.ctime()}"
    #sizes
    sizes_dim = ncfile.createDimension('sizes', len(sizes_labels))
    sizes = ncfile.createVariable('sizes', np.float32, ('sizes',))
    sizes.units = f"µm"
    sizes.long_name = 'size classes of dust particles'
    sizes[:] = np.array([float(x) for x in sizes_labels])
    # Time
    time_dim = ncfile.createDimension('time', None)
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = f"Seconds since {str(basetime)}{timezone}"
    time.long_name = 'time'
    time[:] = np.array([(t - basetime).total_seconds()
                        for t in dustdata['date_time']])
    # PM1
    PM1 = ncfile.createVariable('PM1',np.float32,('time',))
    PM1.units = 'µg/m³'
    PM1.long_name = 'dust concentration up to 1 µm size'
    PM1[:] = dustdata['PM1'].to_numpy()
    # PM2.5
    PM2_5 = ncfile.createVariable('PM2.5',np.float32,('time',))
    PM2_5.units = 'µg/m³'
    PM2_5.long_name = 'dust concentration up to 2.5 µm size'
    PM2_5[:] = dustdata['PM2.5'].to_numpy()
    # PM4
    PM4 = ncfile.createVariable('PM4',np.float32,('time',))
    PM4.units = 'µg/m³'
    PM4.long_name = 'dust concentration up to 4 µm size'
    PM4[:] = dustdata['PM4'].to_numpy()
    # PM10
    PM10 = ncfile.createVariable('PM10',np.float32,('time',))
    PM10.units = 'µg/m³'
    PM10.long_name = 'dust concentration up to 10 µm size'
    PM10[:] = dustdata['PM10'].to_numpy()
    # PMtot
    PMtot = ncfile.createVariable('PMtot',np.float32,('time',))
    PMtot.units = 'µg/m³'
    PMtot.long_name = 'total dust concentration'
    PMtot[:] = dustdata['PMtot'].to_numpy()
    # Cn
    Cn = ncfile.createVariable('Cn',np.float32,('time',))
    Cn.units = 'particles/cm³'
    Cn.long_name = 'count number per volume'
    Cn[:] = dustdata['Cn'].to_numpy()
    # relative humidity
    rH = ncfile.createVariable('rH',np.float32,('time',))
    rH.units = '%'
    rH.long_name = 'relative humidity'
    rH[:] = dustdata['rH'].to_numpy()
    # Dewpoint temperature
    dewT = ncfile.createVariable('dewT',np.float32,('time',))
    dewT.units = '°C'
    dewT.long_name = 'dew point temperature'
    dewT[:] = dustdata['T_dew_point'].to_numpy()
    # temperature
    T = ncfile.createVariable('T',np.float32,('time',))
    T.units = '°C'
    T.long_name = 'air temperature'
    T[:] = dustdata['T'].to_numpy()
    # pressure
    p = ncfile.createVariable('p',np.float32,('time',))
    p.units = 'hPa'
    p.long_name = 'atmospheric pressure'
    p[:] = dustdata['p'].to_numpy()
    # wind speed
    Ws = ncfile.createVariable('Wspeed',np.float32,('time',))
    Ws.units = 'Km/h'
    Ws.long_name = 'wind speed'
    Ws[:] = dustdata['wind speed'].to_numpy()
    # wind direction
    Wd = ncfile.createVariable('Wdir',np.float32,('time',))
    Wd.units = '°'
    Wd.long_name = 'wind direction'
    Wd[:] = dustdata['wind direction'].to_numpy()
    # wind signal quality
    Wq = ncfile.createVariable('Wq',np.float32,('time',))
    Wq.units = '%'
    Wq.long_name = 'wind signal quality'
    Wq[:] = dustdata['wind signal quality'].to_numpy()
    # precipitation intensity
    prec = ncfile.createVariable('prec',np.float32,('time',))
    prec.units = 'l/m²/h'
    prec.long_name = 'precipitation intensity'
    prec[:] = dustdata['prec. int.'].to_numpy()
    # precipitation type
    pt = ncfile.createVariable('ptype',np.int16,('time',))
    pt.units = ''
    pt.long_name = 'precipitation type'
    pt[:] = dustdata['prec. type'].to_numpy().astype(np.int16)
    # size spectra
    spectra = ncfile.createVariable('spectra',np.float32,('time', 'sizes'))
    spectra.units = 'particles/cm³'
    spectra.long_name = 'size spectra: count number per volume in given size channel'
    spectra[:] = dustdata[sizes_labels].to_numpy()
    # flowrate
    fl = ncfile.createVariable('flowrate',np.float32,('time',))
    fl.units = 'l/min'
    fl.long_name = 'flowrate'
    fl[:] = dustdata['flowrate'].to_numpy()
    # velocity
    v = ncfile.createVariable('velocity',np.float32,('time',))
    v.units = 'm/s'
    v.long_name = 'velocity'
    v[:] = dustdata['velocity'].to_numpy()
    # coincidence
    co = ncfile.createVariable('coincidence',np.float32,('time',))
    co.units = '%'
    co.long_name = 'coincidence'
    co[:] = dustdata['coincidence'].to_numpy()
    # pump output
    po = ncfile.createVariable('po',np.float32,('time',))
    po.units = '%'
    po.long_name = 'pump output'
    po[:] = dustdata['pump output'].to_numpy()
    # IADS Temperature
    IADS_T = ncfile.createVariable('IADS_T',np.float32,('time',))
    IADS_T.units = '°C - 318=not activated'
    IADS_T.long_name = 'IADS Temperature'
    IADS_T[:] = dustdata['IADS T'].to_numpy()
    # channel deviation
    cd = ncfile.createVariable('cd',np.float32,('time',))
    cd.units = 'raw channels'
    cd.long_name = 'channel deviation'
    cd[:] = dustdata['channel deviation'].to_numpy()
    # LED Temperature
    LED_T = ncfile.createVariable('LED_T',np.float32,('time',))
    LED_T.units = '°C'
    LED_T.long_name = 'LED Temperature'
    LED_T[:] = dustdata['LED T'].to_numpy()
    # Errors
    errors = ncfile.createVariable('errors',np.uint8,('time',))
    errors.units = ''
    errors.long_name = 'errors: 1=flow; 2=coincidence; 4=pump; 8=weather station; 16=IADS; 32=channel deviation; 64=LED temperature; 128=operation mode'
    errors[:] = np.dot(dustdata[dustdata.columns[11:19]].to_numpy(),
                       np.array([[1],[2],[4],[8],[16],[32],[64],[128]])
                       ).astype(np.uint8)
    # Operation mode
    mode = ncfile.createVariable('mode',np.uint8,('time',))
    mode.units = ''
    mode.long_name = 'operation mode: 0=scope; 1=auto; 2=manual; 3=idle; 4=calib; 5=offset'
    mode[:] = dustdata['modus'].to_numpy().astype(np.uint8)
    # PM1-ambient
    PM1a = ncfile.createVariable('PM1a',np.float32,('time',))
    PM1a.units = 'µg/m³'
    PM1a.long_name = 'PM1 - ambient'
    PM1a[:] = dustdata['alt. PM#1'].to_numpy()
    # PM2.5-ambient
    PM2_5a = ncfile.createVariable('PM2.5a',np.float32,('time',))
    PM2_5a.units = 'µg/m³'
    PM2_5a.long_name = 'PM2.5 - ambient'
    PM2_5a[:] = dustdata['alt. PM#2'].to_numpy()
    # PM4-ambient
    PM4a = ncfile.createVariable('PM4a',np.float32,('time',))
    PM4a.units = 'µg/m³'
    PM4a.long_name = 'PM4 - ambient'
    PM4a[:] = dustdata['alt. PM#3'].to_numpy()
    # PM10-ambient
    PM10a = ncfile.createVariable('PM10a',np.float32,('time',))
    PM10a.units = 'µg/m³'
    PM10a.long_name = 'PM10 - ambient'
    PM10a[:] = dustdata['alt. PM#4'].to_numpy()
    # PMtot-classic
    PMtota = ncfile.createVariable('PMtota',np.float32,('time',))
    PMtota.units = 'µg/m³'
    PMtota.long_name = 'PMtot - ambient'
    PMtota[:] = dustdata['alt. PM#5'].to_numpy()
    # PM1-classic
    PM1c = ncfile.createVariable('PM1c',np.float32,('time',))
    PM1c.units = 'µg/m³'
    PM1c.long_name = 'PM1 - classic'
    PM1c[:] = dustdata['alt. PM#6'].to_numpy()
    # PM2.5-classic
    PM2_5c = ncfile.createVariable('PM2.5c',np.float32,('time',))
    PM2_5c.units = 'µg/m³'
    PM2_5c.long_name = 'PM2.5 - classic'
    PM2_5c[:] = dustdata['alt. PM#7'].to_numpy()
    # PM4-classic
    PM4c = ncfile.createVariable('PM4c',np.float32,('time',))
    PM4c.units = 'µg/m³'
    PM4c.long_name = 'PM4 - classic'
    PM4c[:] = dustdata['alt. PM#8'].to_numpy()
    # PM10-classic
    PM10c = ncfile.createVariable('PM10c',np.float32,('time',))
    PM10c.units = 'µg/m³'
    PM10c.long_name = 'PM10 - classic'
    PM10c[:] = dustdata['alt. PM#9'].to_numpy()
    # PMtot-classic
    PMtotc = ncfile.createVariable('PMtotc',np.float32,('time',))
    PMtotc.units = 'µg/m³'
    PMtotc.long_name = 'PMtot - classic'
    PMtotc[:] = dustdata['alt. PM#10'].to_numpy()
    # PM thoracic
    PMth = ncfile.createVariable('PMth',np.float32,('time',))
    PMth.units = 'µg/m³'
    PMth.long_name = 'PM thoracic'
    PMth[:] = dustdata['alt. PM#11'].to_numpy()
    # PM alveolar
    PMal = ncfile.createVariable('PMal',np.float32,('time',))
    PMal.units = 'µg/m³'
    PMal.long_name = 'PM alveolar'
    PMal[:] = dustdata['alt. PM#12'].to_numpy()
    # PM respirable
    PMre = ncfile.createVariable('PMre',np.float32,('time',))
    PMre.units = 'µg/m³'
    PMre.long_name = 'PM respirable'
    PMre[:] = dustdata['alt. PM#13'].to_numpy()
    #
    ncfile.close()
    os.rename(f"{local_path}/temporary_dust_file.nc",
              f"{local_path}/{filename[:-4]}.nc")


#---------------------------------------------------------------------
# Main
#
while True:
    try:
        start_time = systime.time()
        print(f"---\nACCESS Fidas 200S data downloader - {systime.ctime()}.")
        ftplink, foldercontent = open_ftp_get_file_list(IP_NUMBER,
                                                        USERNAME,
                                                        PASSWORD,
                                                        DATADIR)
        print("Connection open.\nFetching file list.")
        to_be_downloaded = files_to_be_downloaded(foldercontent, TARGET_PATH)
        if to_be_downloaded is None:
            raise ValueError(f"Data files not found in folder: {DATADIR}")
        for fname in to_be_downloaded:
            print(f"Downloading: {fname}")
            download_data_file(ftplink, TARGET_PATH, fname)
            print(f"Converting {fname} to netCDF4")
            convert_to_netCFD4(TARGET_PATH, fname)
        ftplink.quit()
        print("All done. Connection closed.")
        elapsed_time = systime.time() - start_time
        systime.sleep(DownloadInterval - elapsed_time)
    except Exception as e:
        print(str(e))

#---------------------------------------------------------------------


