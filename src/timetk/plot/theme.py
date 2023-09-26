from plotnine import *


def theme_tq(base_size = 11, base_family = ['Arial', 'Helvetica', 'sans-serif'], dpi = 100):
    
    # Tidyquant colors
    blue  = "#2c3e50"
    green = "#18BC9C"
    white = "#FFFFFF"
    grey  = "#cccccc"
    
    return theme(
        
        # Base Inherited Elements
        line = element_line(color = blue, size = 0.5,  lineend="butt"),
        rect = element_rect(fill = white, colour = blue, size = 0.5),
        text = element_text(family = base_family, face = "plain", color = blue, size = base_size, lineheight = 0.9, hjust = 0.5, vjust = 0.5, angle = 0,  margin = dict()),
        
        # Axes
        axis_line=element_blank(),
        axis_text=element_text(size=base_size * 0.6),
        axis_ticks=element_line(color=grey, size=0.5),
        axis_title=element_text(size=base_size*1),
        
        axis_text_y=element_text(ha='center', margin=dict(r=25, l=0)),
        
        # Panel
        panel_background=element_rect(fill = white, color = None),
        panel_border=element_rect(fill = None, color = blue, size = 0.5),
        panel_grid_major=element_line(color = grey, size = 0.33),
        panel_grid_minor=element_line(color = grey, size = 0.33),
        panel_grid_minor_x=element_blank(),
        panel_spacing=0.005,
        
        # Legend
        legend_key=element_rect(color = white),
        legend_position="bottom",
        legend_box=element_rect(fill = None, color = None, size = 0.5, linetype=None),
        legend_text=element_text(size=base_size*0.6, color = blue, margin=dict(t=0, b=0, r=5, l=5)),
        # legend_title=element_text(size=base_size*0.7, color = blue, margin=dict(t=0, b=5, r=5, l=5)),
        legend_title=element_blank(),
        legend_background=element_blank(),
        # legend_box_spacing=0.25,
        
        # Strip
        strip_background=element_rect(fill = blue, color = blue),
        strip_text=element_text(size=base_size*0.8, color = white, margin=dict(t=5, b=5)),
        
        # Plot
        plot_title=element_text(size=base_size*1.2, color = blue, margin=dict(t = 0, r = 0, b = 4, l = 0), face="bold"),
        plot_subtitle=element_text(size=base_size*0.9, color = blue, margin=dict(t = 0, r = 0, b = 3, l = 0)),
        plot_margin=0.025,
        
        dpi=dpi,
        # complete=True
    )
    
    
def palette_light():
    return dict(
        blue         = "#2c3e50", # blue
        red          = "#e31a1c", # red
        green        = "#18BC9C", # green
        yellow       = "#CCBE93", # yellow
        steel_blue   = "#a6cee3", # steel_blue
        navy_blue    = "#1f78b4", # navy_blue
        light_green  = "#b2df8a", # light_green
        pink         = "#fb9a99", # pink
        light_orange = "#fdbf6f", # light_orange
        orange       = "#ff7f00", # orange
        light_purple = "#cab2d6", # light_purple
        purple       = "#6a3d9a"  # purple
    )