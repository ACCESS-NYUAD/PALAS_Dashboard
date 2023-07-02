# %%
#-------------------------------------------------------------------------------
#
# Run this with: 
#     voila --VoilaConfiguration.file_whitelist="['DUSTMONITOR*\.(nc|txt)', 'ACCESS.png', 'CITIES.png']"  Fidas_dashboard.ipynb
#
# Without whitelisting the logos in the 'about' tab won't appear, and the data file download  
# will not work (error 403 'forbidden').
# Note that within Jupyter notebook/lab only text file download will work (will be opened and
# visualized in a new tab). But Jupyter doesn't know how to handle .nc files, and so gives you
# a pop-up error with a silly message ('File download error: the file is not utf-8 encoded').
# Download will work in voilà, provided that the files have been correctly whitelisted.
# In the next cell the calls to 'display' may be commented out when working in Jupyter: they are
# meant to avoid excessive whitespace on the page margins when running the dashboard in voilà.
#

# %%
# %matplotlib ipympl
import matplotlib.pyplot as plt
from IPython.display import display, HTML, FileLink
# display(HTML("<style>.jp-Cell {padding: 0 !important; }</style>"))
# display(HTML("<style>.jp-Notebook {padding: 0 !important; }</style>"))
from netCDF4 import Dataset
import datetime
import dateutil
from fnmatch import fnmatch
import ipywidgets as widgets
import os
import numpy as np
DATADIRECTORY = '.'

# %%
class DataStore():
    no_show = ['PM1a', 'PM2.5a', 'PM4a', 'PM10a', 'PMtota', 
               'PM1c', 'PM2.5c', 'PM4c', 'PM10c', 'PMtotc',
               'PMth', 'PMal', 'PMre']
    indep_var = ['sizes', 'time']
    multidim_var = ['spectra']
    # Outliers in the following variables should not be removed
    do_not_clean = ['errors', 'mode', 'ptype', 'cd', 'po', 'coincidence']
    iDATA      = 0
    iLONG_NAME = 1
    iUNITS     = 2
    def __init__(self):
        ls = os.listdir(DATADIRECTORY)
        self.ncfiles = [fn for fn in ls if fnmatch(fn, 'DUSTMONITOR*.nc')]
        self.ncfiles = sorted(self.ncfiles, key=lambda name: name[-10:-3])
        self.txtfiles = [fn for fn in ls if fnmatch(fn, 'DUSTMONITOR*.txt')]
        self.txtfiles = sorted(self.txtfiles, key=lambda name: name[-10:-3])
        self.data = {}
        for fn in self.ncfiles:
            ncdata = Dataset(fn)
            for key in ncdata.variables.keys():
                if key in self.no_show:
                    continue
                if not key in self.data:
                    #-------------    iDATA,                    iLONG_NAME,            iUNITS
                    self.data[key] = [np.array(ncdata[key][:]), ncdata[key].long_name, ncdata[key].units]
                else:
                    if key!='sizes':
                        self.data[key][self.iDATA] = np.concatenate((self.data[key][self.iDATA], 
                                                                     np.array(ncdata[key][:])))
            ncdata.close()

# %%
data = DataStore()

# %%
class ButtonList():
    maxpressed = 4
    def __init__(self, data_store_instance, plotter_instance):
        self.store = data_store_instance
        self.plotter = plotter_instance
        time_series = list(self.store.data.keys())
        for series in self.store.indep_var + self.store.multidim_var:
            time_series.remove(series)
        # To keep track of the line colors in the plots of the pressed buttons
        #the callback adds an attribute to the button widgets
        self.available_colors = [3, 2, 1, 0]
        button_list = []
        for ts in time_series:
            button_list.append(
                widgets.ToggleButton(
                    value=False,
                    description=ts,
                    tooltip=f"{self.store.data[ts][self.store.iLONG_NAME]} ({self.store.data[ts][self.store.iUNITS]})",
                    disabled=False,
                )
            )
            button_list[-1]._Fidas_dashboard_units = self.store.data[ts][self.store.iUNITS]
            #At the beginning show the graph of PM2.5
            if ts=='PM2.5':
                button_list[-1].value=True
                button_list[-1]._Fidas_dashboard_color = self.available_colors.pop()
                self.active_buttons = [button_list[-1]]
        self.pressed = 1
        self.buttons = widgets.VBox(button_list)
        for bt in self.buttons.children:
            bt.observe(self.callback)
        
    def callback(self, wdic):
        if wdic['name']=='value':
            if wdic['owner'].value==True:
                self.pressed += 1
                if self.pressed <= 4:
                    wdic['owner']._Fidas_dashboard_color = self.available_colors.pop()
            else:
                self.pressed -= 1
                if self.pressed < 4:
                    self.available_colors.append(wdic['owner']._Fidas_dashboard_color)
                    self.available_colors.sort(reverse=True)
            if self.pressed > self.maxpressed:
                #---maybe add here a popup stating one can't press more than 4 buttons ---#
                #setting to False triggers another callback
                #which takes care of the counter
                wdic['owner'].value=False
            self.active_buttons = [b for b in self.buttons.children if b.value==True]
            self.plotter.plot_callback()

# %%
def get_grid_status(axis):
    gridx = any([line.get_visible() for line in axis.get_xgridlines()])
    gridy = any([line.get_visible() for line in axis.get_ygridlines()])
    return gridx, gridy

def set_grid_status(status, axis):
    axis.grid(visible=status[0], axis='x')
    axis.grid(visible=status[1], axis='y')

class Plotter():
    linecolors = {0: 'tab:blue',
                  1: 'tab:brown',
                  2: 'tab:orange',
                  3: 'tab:olive'}
    def __init__(self, DataStore_instance):
        self.store = DataStore_instance
        plt.ioff()
        self.fig = plt.figure()
        self.fig.canvas.header_visible = False
        self.fig.canvas.resizable = False
        self.fig.canvas.toolbar_position = 'right'
        self.fig.canvas.layout.width = '100%'
        self.fig.set_figwidth(7)
        basetime = dateutil.parser.parse(self.store.data['time'][self.store.iUNITS].split(' ')[-1])
        self.times = np.array([basetime + datetime.timedelta(seconds=x) 
                                for x in self.store.data['time'][self.store.iDATA]])
        self.clean_data = True
        self.axes = []
        #compute indexes of non-zero errors. Some data appear to be wrong even 
        #immediately before or after an error. Thus, first I compute a moving average
        #of the errors time series, then I extract the index of the non-zero averaged error array
        avgd_err = self.store.data['errors'][self.store.iDATA].copy()
        avgd_err[1:-1] = (avgd_err[2:] + avgd_err[1:-1] + avgd_err[:-2])/3
        self.error_indexes = np.where(avgd_err > 0)[0]
    
    def register_date_range_slider(self, slider):
        self.date_range_slider = slider
        
    def register_button_list(self, button_list):
        self.buttons = button_list
        
    def plot_callback(self):
        if not hasattr(self, 'date_range_slider'):
            raise AttributeError("The date range slider has not been registered in the Plotter instance")
        if not hasattr(self, 'buttons'):
            raise AttributeError("The button list has not been registered in the Plotter instance")
        old_axes_and_grid_status = [(ax._Fidas_dashboard_description, get_grid_status(ax)) 
                                    for ax in self.axes]
        self.fig.clf()
        self.axes = []
        self.ax = self.fig.add_subplot()
        for i, b in enumerate(self.buttons.active_buttons):
            if i==0:
                ax = self.ax
            else:
                ax = self.ax.twinx()
            ax._Fidas_dashboard_description = b.description
            self.axes.append(ax)
            if i == 1:
                ax.spines['right'].set_position(('outward', 20))
            if i == 2:
                ax.spines['right'].set_position(('outward', 50))
            if i == 3:
                ax.spines['right'].set_position(('outward', 100))
            tbp = self.store.data[b.description][self.store.iDATA].copy()
            if self.clean_data and not b.description in self.store.do_not_clean:
                tbp[self.error_indexes] = np.nan
            clr = self.linecolors[b._Fidas_dashboard_color]
            ax.plot(self.times, tbp, '.', 
                    markersize=1, color=clr)
            ax.set_ylabel(f"{b.description}   ({b._Fidas_dashboard_units})", 
                          fontsize=12, color=clr)
            ax.tick_params(axis='y', colors=clr)
            for old_ax_descr, grid_status in old_axes_and_grid_status:
                if old_ax_descr==ax._Fidas_dashboard_description:
                    set_grid_status(grid_status, ax)
        self.fig.autofmt_xdate(rotation=45)
        self.date_range_callback({'name': 'value'}) #faking a slider change event
        
    def width_callback(self, wdic):
        if wdic['name']=='value':
            self.fig.set_figwidth(wdic['owner'].value)
        self.plot_callback()
        
    def clean_callback(self, wdic):
        if wdic['name']=='value':
            self.clean_data = wdic['owner'].value
        self.plot_callback()
        
    def date_range_callback(self, wdic):
        if wdic['name']=='value' and self.buttons.active_buttons != []:
            #The right end of the date range needs to be rounded up to the next day
            min_day = self.date_range_slider.value[0]
            max_day = self.date_range_slider.value[1] + datetime.timedelta(days=1)
            self.ax.set_xlim((min_day, max_day))
        if wdic['name']=='value':
            self.fig.tight_layout(pad=1.02)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            

# %%
plotter = Plotter(data)
button_list = ButtonList(data, plotter)
plotter.register_button_list(button_list)

# %%
clean_data = widgets.ToggleButton(value=plotter.clean_data, 
                                  description='Clean Data',
                                  tooltip="If clicked, data with error flags are not visualized",
                                  disabled=False,
                                 )
clean_data.observe(plotter.clean_callback)

# %%
fig_width_slider = widgets.FloatSlider(value=7.0, 
                                       min=1, 
                                       max=15.0,
                                       step=0.1,
                                       description='Figure width:',
                                       disabled=False,
                                       continuous_update=False,
                                       orientation='horizontal',
                                       readout=True,
                                       readout_format='.1f',
                                       tooltip='Change to resize the plotting area width'
                                      )
fig_width_slider.observe(plotter.width_callback)

# %%
slider_days = np.unique([x.date() for x in plotter.times])
date_range_slider = widgets.SelectionRangeSlider(
    options = slider_days,
    description = 'Date range:',
    orientation = 'horizontal',
    index = (0, len(slider_days)-1),
    disabled = False,
    continuous_update = False,
    tooltip = 'Select the date range to be plotted',
    layout=widgets.Layout(width='100%')
)
date_range_slider.observe(plotter.date_range_callback)
plotter.register_date_range_slider(date_range_slider)
plotter.plot_callback()

# %%
decorated_canvas = widgets.VBox([widgets.HBox([clean_data, fig_width_slider]),
                                 date_range_slider,
                                 plotter.fig.canvas])
tab_time_series = widgets.HBox([button_list.buttons, decorated_canvas])

# %%
#-------------------------------------------------------------------------------------------------------------
#***Classes and widgets for the particle spectra tab***

# %%
class SpectraPlotter():
    def __init__(self, DataStore_instance):
        self.store = DataStore_instance
        plt.ioff()
        self.fig = plt.figure()
        self.fig.canvas.header_visible = False
        self.fig.canvas.resizable = False
        self.fig.canvas.toolbar_position = 'right'
        self.fig.canvas.layout.width = '100%'
        self.fig.set_figwidth(8)
        self.fig.set_figheight(5)
        basetime = dateutil.parser.parse(self.store.data['time'][self.store.iUNITS].split(' ')[-1])
        self.times = np.array([basetime + datetime.timedelta(seconds=x) 
                                for x in self.store.data['time'][self.store.iDATA]])
        self.plot_callback(len(self.times)-1)
        
    def plot_callback(self, i):
        self.fig.clf()
        self.ax = self.fig.add_subplot()
        self.ax.step(
            self.store.data['sizes'][self.store.iDATA],
            self.store.data['spectra'][self.store.iDATA][i,:])
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        self.ax.set_xlabel(
            f"Size ({self.store.data['sizes'][self.store.iUNITS]})",
            fontsize=14)
        self.ax.set_ylabel(
            f"Particle Count ({self.store.data['spectra'][self.store.iUNITS]})",
            fontsize=14)
        self.ax.set_ylim(1.e-3, 1.e3)
        self.ax.grid(True)
        self.fig.tight_layout(pad=1.)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# %%
spectra_plotter = SpectraPlotter(data)

# %%
class DateTimeSelector():
    n_slider_steps = 700
    def __init__(self, SpectraPlotter_instance):
        self.plotter = SpectraPlotter_instance
        self.slider = widgets.SelectionSlider(
            options=np.arange(self.n_slider_steps+1),
            value=self.n_slider_steps,
            description='Select date:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            tooltip = 'Use the arrows for fine control',
            layout={'width': '50%', 'padding': '0px 10px 0px 0px'},
            readout=False)
        self.slider_active = True
        self.itime = len(self.plotter.times)-1
        self.slider.observe(self.slider_callback)
        self.slider_label = widgets.Label(
            value=str(self.plotter.times[-1]),
            layout={'padding': '0px 10px 0px 10px'})
        self.forward = widgets.ToggleButton(
            value=False,
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Forward 1 minute',
            layout={'width': 'max-content'},
            icon='angle-right')
        self.forward.observe(self.forward_callback)
        self.backward = widgets.ToggleButton(
            value=False,
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Backward 1 minute',
            layout={'width': 'max-content'},
            icon='angle-left')
        self.backward.observe(self.backward_callback)
        self.fast_forward = widgets.ToggleButton(
            value=False,
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Forward 1 hour',
            layout={'width': 'max-content'},
            icon='angle-double-right')
        self.fast_forward.observe(self.fast_forward_callback)
        self.fast_backward = widgets.ToggleButton(
            value=False,
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Backward 1 hour',
            layout={'width': 'max-content'},
            icon='angle-double-left')
        self.fast_backward.observe(self.fast_backward_callback)
        self.ffast_forward = widgets.ToggleButton(
            value=False,
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Forward 1 day',
            layout={'width': 'max-content'},
            icon='arrow-right')
        self.ffast_forward.observe(self.ffast_forward_callback)
        self.ffast_backward = widgets.ToggleButton(
            value=False,
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Backward 1 day',
            layout={'width': 'max-content'},
            icon='arrow-left')
        self.ffast_backward.observe(self.ffast_backward_callback)
        self.widget = widgets.HBox(
            [self.slider, 
             self.slider_label,
             self.ffast_backward,
             self.fast_backward,
             self.backward, 
             self.forward,
             self.fast_forward,
             self.ffast_forward],
            layout=widgets.Layout(width='900px'))

    def forward_callback(self, wdic):
        self.button_callback(wdic, +1)
    
    def backward_callback(self, wdic):
        self.button_callback(wdic, -1)

    def fast_forward_callback(self, wdic):
        self.button_callback(wdic, +60)
    
    def fast_backward_callback(self, wdic):
        self.button_callback(wdic, -60)
            
    def ffast_forward_callback(self, wdic):
        self.button_callback(wdic, +60*24)
    
    def ffast_backward_callback(self, wdic):
        self.button_callback(wdic, -60*24)
            
    def button_callback(self, wdic, noffset):
        if (wdic['name']=='value' and 
            wdic['owner'].value): 
            self.itime += noffset
            if self.itime < 0:
                self.itime = 0
            if self.itime >= len(self.plotter.times):
                self.itime = len(self.plotter.times)-1
            self.slider_label.value = str(self.plotter.times[self.itime])
            self.n_step = int((self.itime*self.n_slider_steps)/(len(self.plotter.times)-1))
            self.slider_active = False
            self.slider.value = self.n_step #this will call slider_callback
            self.slider_active = True
            wdic['owner'].value = False #Don't keep the button on
            self.plotter.plot_callback(self.itime)

    def slider_callback(self, wdic):
        if (wdic['name']=='value' and
            self.slider_active): #slider is not active if this is called by a button press
            self.n_step = wdic['owner'].value
            self.itime = int((self.n_step/self.n_slider_steps)*(len(self.plotter.times)-1))
            self.slider_label.value = str(self.plotter.times[self.itime])
            self.plotter.plot_callback(self.itime)
        

datetime_slider = DateTimeSelector(spectra_plotter)

# %%
tab_spectra = widgets.VBox([datetime_slider.widget,
                            spectra_plotter.fig.canvas])

# %%
#----------------------------------------------------------------------------------------------
#***Widgets for the file download tab***

# %%
ncfiles_links = [
    widgets.HTML(
        value="<b>Data in netCDF4 format</b><br>"
    )
] + [
    widgets.HTML(
        value='<u style="color:blue;">'+FileLink(x)._repr_html_()+'</u>',
        placeholder='',
        description='',
        tooltip='Click the link to download the file'
    ) for x in data.ncfiles
]
ncfiles_box = widgets.VBox(
    ncfiles_links,
    layout=widgets.Layout(
        margin='0 100px 0 100px'
    )
)

# %%
txtfiles_links =  [
    widgets.HTML(
        value="<b>Data in tabbed text format</b><br>"
    )
] + [
    widgets.HTML(
        value='<u style="color:blue;">'+FileLink(x)._repr_html_()+'</u>',
        placeholder='',
        description='',
        tooltip='Click the link to download the file'
    ) for x in data.txtfiles
]
txtfiles_box = widgets.VBox(
    txtfiles_links,
    layout=widgets.Layout(
        margin='0 100px 0 100px'
    )
)

# %%
tab_downloads = widgets.HBox([ncfiles_box, txtfiles_box])

# %%
#----------------------------------------------------------------------------------------------
#***Widgets for the intro/about tab***

# %%
intro = widgets.HTML(
    value="""<p style="line-height: 150%">A Palas Fidas 200S aerosol spectrometer is operated at NYUAD by the 
    Arabian Center for Climate and Environmental Sciences, jointly with the Center 
    for Interacting Urban Networks. From the tabs above you can visualize current 
    and past measurements of dust concentration, as well as basic meteorological 
    parameters. You can also download the monthly data in netCDF4 or tabbed text format.</p>
    <p>&nbsp;</p>""",
    layout=widgets.Layout(width='700px')
)
logo_ACCESS = widgets.HTML(
    value='<img src="ACCESS.png" alt="Arabian Center for Climate and Environmental Sciences" style="width:300px">',
    layout=widgets.Layout(
        margin='0 20px 0 20px'
    )
)
logo_CITIES = widgets.HTML(
    value='<img src="CITIES.png" alt="Center for Interacting Urban Networks" style="width:300px">',
    layout=widgets.Layout(
        margin='0 20px 0 20px'
    )
)
tab_about = widgets.VBox([intro, widgets.HBox([logo_ACCESS, logo_CITIES])])

# %%
#----------------------------------------------------------------------------------------------
#***Display the tabbed interface***

# %%

tabbed_interface = widgets.Tab()
tabbed_interface.children = [tab_about, tab_time_series, tab_spectra, tab_downloads]
tabbed_interface.set_title(0, 'About')
tabbed_interface.set_title(1, 'Time series')
tabbed_interface.set_title(2, 'Particle spectra')
tabbed_interface.set_title(3, 'Data download')
display(tabbed_interface)

# %%



