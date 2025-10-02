# from framework.config.data_config

BORDER_COLORS = {"ILM": "#ff0000", "NFL/GCL": "#ffff00", "GCL/IPL": "#054aea",
                 "IPL/INL": "#ffc000", "INL/OPL": "#00ff00", "OPL/ONL": "#ad89f1",
                 "MZ/EZ": "#0ebc0e", "OS/RPE": "#00ffff", "RPE/BM": "#0000ff", "BM": "#ae0491",
                 "CSI": "#ff6400", "ASL": "#ff0000", "PSL": "#ff0000", "ELM": "#f0ca95",
                 "EZ/OS": "#af76de", "IZ/RPE": "#00ffff"}

BORDER_COLORS = {
    **BORDER_COLORS,
    "Anterior Surface": BORDER_COLORS["ILM"],
    "Epithelium": BORDER_COLORS["NFL/GCL"],
    "Endothelium": BORDER_COLORS["RPE/BM"],
    "Iris Anterior Surface": BORDER_COLORS["IPL/INL"],
    "Iris Posterior Surface": BORDER_COLORS["IZ/RPE"],
    "Lens Anterior Surface": BORDER_COLORS["OPL/ONL"],
    "Lens Posterior Surface": BORDER_COLORS["MZ/EZ"]
}

LAYER_COLORS = {"TOP": "#7d67b5", **BORDER_COLORS}

BORDER_ORDER = ["ILM", "NFL/GCL", "GCL/IPL",
                "IPL/INL", "INL/OPL", "OPL/ONL",
                "ELM", "MZ/EZ", "EZ/OS", "IZ/RPE", "OS/RPE", "RPE/BM",
                "BM", "CSI", "ASL", "PSL",
                'Anterior Surface', 'Epithelium', 'Endothelium', 'Iris Anterior Surface', 'Iris Posterior Surface', 'Lens Anterior Surface', 'Lens Posterior Surface']

POSTERIOR_BORDER_ORDER = ['ILM', 'NFL/GCL', 'GCL/IPL', 'IPL/INL', 'INL/OPL', 'OPL/ONL',
                          'ELM', 'MZ/EZ', 'EZ/OS', 'IZ/RPE', 'RPE/BM', 'BM', 'CSI']

ANTERIOR_FR_BORDER_ORDER = ['Anterior Surface', 'Epithelium', 'Endothelium',
                            'Iris Anterior Surface', 'Iris Posterior Surface',
                            'Lens Anterior Surface', 'Lens Posterior Surface']

ANY_BORDER_THRESH_NAME = "ALL"
