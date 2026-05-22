import os
import json
import numpy as np
import nibabel as nib
import scipy.stats as stats
import dash
from dash import html, dcc, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go

# Define paths
BASE_DIR = os.environ.get("PSYSYNTH_BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load background brain template
brain_path = os.path.join(BASE_DIR, "data/anorexia/functional/MNI152_T1_2mm_brain.nii.gz")
try:
    print(f"Loading anatomical brain template from: {brain_path}")
    brain_img = nib.load(brain_path)
    brain_data = brain_img.get_fdata()
    affine = brain_img.affine
    inv_affine = np.linalg.inv(affine)
except Exception as e:
    print(f"Error loading anatomical template: {e}")
    # Fallback template dimensions
    brain_data = np.zeros((91, 109, 91))
    affine = np.diag([-2.0, 2.0, 2.0, 1.0]) # Standard 2mm spacing
    affine[0, 3] = 90.0
    affine[1, 3] = -126.0
    affine[2, 3] = -72.0
    inv_affine = np.linalg.inv(affine)

# Load modalities maps
maps = {
    'structure': {
        'exceedance': None,
        'posterior_mean': None
    },
    'function': {
        'exceedance': None,
        'posterior_mean': None
    }
}

map_paths = {
    ('structure', 'exceedance'): os.path.join(BASE_DIR, "data/anorexia/structure_thresholded_exceedance_map.nii.gz"),
    ('structure', 'posterior_mean'): os.path.join(BASE_DIR, "data/anorexia/structure_thresholded_posterior_mean.nii.gz"),
    ('function', 'exceedance'): os.path.join(BASE_DIR, "data/anorexia/function_thresholded_exceedance_map.nii.gz"),
    ('function', 'posterior_mean'): os.path.join(BASE_DIR, "data/anorexia/function_thresholded_posterior_mean.nii.gz"),
}

for (modality, map_type), path in map_paths.items():
    if os.path.exists(path):
        try:
            maps[modality][map_type] = nib.load(path).get_fdata()
            print(f"Successfully loaded {modality} {map_type} map from {path}")
        except Exception as e:
            print(f"Error loading map {path}: {e}")
            maps[modality][map_type] = np.zeros_like(brain_data)
    else:
        print(f"Warning: Map not found at {path}. Initializing with zeros.")
        maps[modality][map_type] = np.zeros_like(brain_data)

# Ingest metadata and calculate empirical prior parameters at startup
priors = {}
metadata = {}

try:
    # Precompute sphere offsets (10mm distance)
    voxel_sizes = nib.affines.voxel_sizes(affine)
    max_offsets = np.ceil(10.0 / voxel_sizes).astype(int)
    rx = np.arange(-max_offsets[0], max_offsets[0] + 1)
    ry = np.arange(-max_offsets[1], max_offsets[1] + 1)
    rz = np.arange(-max_offsets[2], max_offsets[2] + 1)
    grid_i, grid_j, grid_k = np.meshgrid(rx, ry, rz, indexing='ij')
    grid_offsets = np.stack([grid_i.ravel(), grid_j.ravel(), grid_k.ravel()], axis=1)
    phys_offsets = grid_offsets @ affine[:3, :3].T
    dists = np.linalg.norm(phys_offsets, axis=1)
    sphere_offsets = grid_offsets[dists <= 10.0]
except Exception as e:
    print(f"Error precomputing sphere offsets: {e}")
    sphere_offsets = np.zeros((1, 3))

shape = brain_data.shape
mask_data = brain_data > 0
total_voxels = int(np.sum(mask_data))

for modality in ['structure', 'function']:
    metadata_path = os.path.join(BASE_DIR, f"data/anorexia/{modality}.json")
    try:
        print(f"Loading {modality} metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            studies = json.load(f)
        print(f"Loaded {len(studies)} {modality} studies.")
    except Exception as e:
        print(f"Error loading {modality} metadata: {e}")
        studies = []
        
    metadata[modality] = studies
    N = len(studies)
    
    if N == 0:
        priors[modality] = {'alpha_0': 0.153213, 'beta_0': 8.390790, 'N': 0}
        continue
        
    try:
        all_maps = np.zeros(shape + (N,), dtype=np.uint8)
        for idx, study in enumerate(studies):
            peaks = study.get('Peaks', [])
            if not peaks:
                continue
            study_voxels = []
            for peak in peaks:
                coord_x, coord_y, coord_z = peak.get('x'), peak.get('y'), peak.get('z')
                if coord_x is None or coord_y is None or coord_z is None:
                    continue
                coord_homg = np.array([coord_x, coord_y, coord_z, 1.0])
                voxel_idx = np.round(inv_affine @ coord_homg)[:3].astype(int)
                if (0 <= voxel_idx[0] < shape[0]) and (0 <= voxel_idx[1] < shape[1]) and (0 <= voxel_idx[2] < shape[2]):
                    sphere_voxels = voxel_idx + sphere_offsets
                    study_voxels.append(sphere_voxels)
            if study_voxels:
                study_voxels = np.concatenate(study_voxels, axis=0)
                in_bounds = (
                    (study_voxels[:, 0] >= 0) & (study_voxels[:, 0] < shape[0]) &
                    (study_voxels[:, 1] >= 0) & (study_voxels[:, 1] < shape[1]) &
                    (study_voxels[:, 2] >= 0) & (study_voxels[:, 2] < shape[2])
                )
                valid_voxels = study_voxels[in_bounds]
                all_maps[valid_voxels[:, 0], valid_voxels[:, 1], valid_voxels[:, 2], idx] = 1
                
        k_map = np.sum(all_maps, axis=3)
        total_hits = int(np.sum(k_map[mask_data]))
        GBR = total_hits / (total_voxels * N)
        W = np.sqrt(N)
        alpha_0 = GBR * W
        beta_0 = (1.0 - GBR) * W
        priors[modality] = {'alpha_0': alpha_0, 'beta_0': beta_0, 'N': N}
        print(f"Empirical Prior Parameters Calculated for {modality}: GBR={GBR:.8f}, alpha_0={alpha_0:.6f}, beta_0={beta_0:.6f}")
    except Exception as e:
        print(f"Error calculating priors for {modality}, using default: {e}")
        priors[modality] = {'alpha_0': 0.153213, 'beta_0': 8.390790, 'N': N}

# Colormap Color Bounds for Gradients
COLORMAPS = {
    'exceedance': {
        'start': (4, 120, 87),    # Dark Emerald
        'end': (52, 211, 153)    # Light Emerald
    },
    'posterior_mean': {
        'start': (109, 40, 217), # Dark Purple
        'end': (192, 132, 252)  # Light Purple
    }
}

# Helper functions for coordinate conversions
def voxel_to_mni(i, j, k):
    coord_homg = np.array([i, j, k, 1.0])
    x, y, z = (affine @ coord_homg)[:3]
    return float(x), float(y), float(z)

def mni_to_voxel(x, y, z):
    coord_homg = np.array([x, y, z, 1.0])
    i, j, k = np.round(inv_affine @ coord_homg)[:3].astype(int)
    # Clip to valid volume ranges
    i = np.clip(i, 0, brain_data.shape[0] - 1)
    j = np.clip(j, 0, brain_data.shape[1] - 1)
    k = np.clip(k, 0, brain_data.shape[2] - 1)
    return int(i), int(j), int(k)

# Image slice blending logic
def generate_rgba_slice(bg_slice, stat_slice, stat_min, stat_max, colormap, alpha=0.85):
    # Normalize background
    bg_min = bg_slice.min()
    bg_max = bg_slice.max()
    if bg_max > bg_min:
        bg_norm = (bg_slice - bg_min) / (bg_max - bg_min)
    else:
        bg_norm = np.zeros_like(bg_slice)
        
    # Scale background to a nice slate-dark grayscale range [0, 80]
    gray = (bg_norm * 80).astype(np.uint8)
    
    H, W = bg_slice.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = 255 # Fully opaque background
    
    # Overlay stats values > 0
    mask = stat_slice > 0
    if np.any(mask):
        val = stat_slice[mask]
        if stat_max > stat_min:
            val_norm = (val - stat_min) / (stat_max - stat_min)
            val_norm = np.clip(val_norm, 0, 1)
        else:
            val_norm = np.ones_like(val)
            
        start_rgb = colormap['start']
        end_rgb = colormap['end']
        
        r_over = (start_rgb[0] + val_norm * (end_rgb[0] - start_rgb[0])).astype(np.uint8)
        g_over = (start_rgb[1] + val_norm * (end_rgb[1] - start_rgb[1])).astype(np.uint8)
        b_over = (start_rgb[2] + val_norm * (end_rgb[2] - start_rgb[2])).astype(np.uint8)
        
        rgba[mask, 0] = (rgba[mask, 0] * (1. - alpha) + r_over * alpha).astype(np.uint8)
        rgba[mask, 1] = (rgba[mask, 1] * (1. - alpha) + g_over * alpha).astype(np.uint8)
        rgba[mask, 2] = (rgba[mask, 2] * (1. - alpha) + b_over * alpha).astype(np.uint8)
        
    # Transpose back to (W, H, 4) for Plotly imshow compatibility (rows=y-axis, cols=x-axis)
    return np.transpose(rgba, (1, 0, 2))

def get_axial_slice(mni, stat, k, stat_min, stat_max, colormap):
    return generate_rgba_slice(mni[:, :, k], stat[:, :, k], stat_min, stat_max, colormap)

def get_coronal_slice(mni, stat, j, stat_min, stat_max, colormap):
    return generate_rgba_slice(mni[:, j, :], stat[:, j, :], stat_min, stat_max, colormap)

def get_sagittal_slice(mni, stat, i, stat_min, stat_max, colormap):
    return generate_rgba_slice(mni[i, :, :], stat[i, :, :], stat_min, stat_max, colormap)

# Query functional study distances
# Query coordinate stats
def query_coordinate_stats(x, y, z, modality='function'):
    k = 0
    contributing_list = []
    
    v_query = np.round(inv_affine @ np.array([x, y, z, 1.0]))[:3]
    
    studies = metadata.get(modality, [])
    alpha_0 = priors[modality]['alpha_0']
    beta_0 = priors[modality]['beta_0']
    N_val = priors[modality]['N']
    
    for study in studies:
        pmid = study.get("PMID", "Unknown")
        sample_size = study.get("sample_size", "N/A")
        study_modality = study.get("modality", "N/A")
        peaks = study.get("Peaks", [])
        
        min_dist = float('inf')
        for peak in peaks:
            px = peak.get("x")
            py = peak.get("y")
            pz = peak.get("z")
            if px is None or py is None or pz is None:
                continue
                
            v_peak = np.round(inv_affine @ np.array([px, py, pz, 1.0]))[:3]
            voxel_diff = v_query - v_peak
            phys_diff = affine[:3, :3] @ voxel_diff
            dist = np.linalg.norm(phys_diff)
            
            if dist < min_dist:
                min_dist = dist
                
        if min_dist <= 10.0:
            k += 1
            contributing_list.append({
                'PMID': pmid,
                'sample_size': sample_size,
                'modality': study_modality,
                'distance': round(min_dist, 2)
            })
            
    # Sort by proximity
    contributing_list.sort(key=lambda s: s['distance'])
    if alpha_0 + beta_0 + N_val > 0:
        post_mean = (alpha_0 + k) / (alpha_0 + beta_0 + N_val)
    else:
        post_mean = 0.0
    return k, post_mean, contributing_list

# Initial values and bounds for dimensions
init_i = brain_data.shape[0] // 2
init_j = brain_data.shape[1] // 2
init_k = brain_data.shape[2] // 2

max_i = brain_data.shape[0] - 1
max_j = brain_data.shape[1] - 1
max_k = brain_data.shape[2] - 1

# Initialize Dash application
app = dash.Dash(
    __name__,
    title="PsySynth Meta-Analysis Portal",
    update_title=None
)

# App Layout
app.layout = html.Div(className='dashboard-container', children=[
    # State Store
    dcc.Store(id='active-coord-store', data={'i': init_i, 'j': init_j, 'k': init_k}),
    
    # Header Section
    html.Div(className='dashboard-header', children=[
        html.Div(className='header-title-container', children=[
            html.H1("PsySynth Meta-Analysis Portal"),
            html.P("Bayesian Neuroimaging Meta-Analysis & Coordinate Query Dashboard")
        ]),
        html.Div(className='header-badge', children=[
            html.Span(className='header-badge-dot'),
            html.Span("Empirical Prior: Active")
        ])
    ]),
    
    # Main Body Layout
    html.Div(className='dashboard-layout', children=[
        # Left Side Control Bar
        html.Div(className='sidebar', children=[
            # Card 1: Configuration Selection
            html.Div(className='dashboard-card', children=[
                html.H3(className='card-title', children=[
                    html.Span("⚙️", className='card-title-icon primary'),
                    "Modalities & Settings"
                ]),
                
                html.Label("Meta-Analysis Modality", className='control-label'),
                dcc.Dropdown(
                    id='modality-dropdown',
                    options=[
                        {'label': 'Functional MRI (BOLD, PET)', 'value': 'function'},
                        {'label': 'Structural MRI (VBM, Volume)', 'value': 'structure'}
                    ],
                    value='function',
                    clearable=False,
                    className='dash-dropdown'
                ),
                
                html.Div(style={'height': '1rem'}),
                
                html.Label("Statistic Mapping", className='control-label'),
                dcc.Dropdown(
                    id='map-type-dropdown',
                    options=[
                        {'label': 'Exceedance Probability (τ=0.10)', 'value': 'exceedance'},
                        {'label': 'Posterior Mean Probability', 'value': 'posterior_mean'}
                    ],
                    value='exceedance',
                    clearable=False,
                    className='dash-dropdown'
                ),
            ]),
            
            # Card 2: Coordinates Lookup Control Panel
            html.Div(className='dashboard-card', children=[
                html.H3(className='card-title', children=[
                    html.Span("📍", className='card-title-icon secondary'),
                    "Coordinate Navigation"
                ]),
                
                html.Label("Physical Space (MNI Coordinates x, y, z)", className='control-label'),
                html.Div(className='coord-input-group', children=[
                    html.Div(className='coord-input-container', children=[
                        dcc.Input(id='input-x', type='number', placeholder='x', className='coord-number-input')
                    ]),
                    html.Div(className='coord-input-container', children=[
                        dcc.Input(id='input-y', type='number', placeholder='y', className='coord-number-input')
                    ]),
                    html.Div(className='coord-input-container', children=[
                        dcc.Input(id='input-z', type='number', placeholder='z', className='coord-number-input')
                    ]),
                ]),
                
                html.Div(style={'height': '1rem'}),
                
                html.Label("Voxel Indices (Matrix dimensions i, j, k)", className='control-label'),
                html.Div(className='coord-input-group', children=[
                    html.Div(className='coord-input-container', children=[
                        dcc.Input(id='input-i', type='number', placeholder='i', className='coord-number-input')
                    ]),
                    html.Div(className='coord-input-container', children=[
                        dcc.Input(id='input-j', type='number', placeholder='j', className='coord-number-input')
                    ]),
                    html.Div(className='coord-input-container', children=[
                        dcc.Input(id='input-k', type='number', placeholder='k', className='coord-number-input')
                    ]),
                ]),
                
                html.Div(style={'height': '1.25rem'}),
                
                html.Button("Go to Coordinates", id='coord-update-btn', className='btn-primary'),
                html.Div(style={'height': '0.5rem'}),
                html.Button("Reset to Center (MNI 0,0,18)", id='reset-btn', className='btn-primary', style={'backgroundColor': '#334155', 'color': '#f8fafc'})
            ]),
            
            # Card 3: Active Location Stats
            html.Div(className='dashboard-card', children=[
                html.H3(className='card-title', children=[
                    html.Span("📊", className='card-title-icon primary'),
                    "Voxel Statistics"
                ]),
                
                html.Div(className='info-row', children=[
                    html.Span("Selected Voxel Indices", className='info-label'),
                    html.Span(id='display-voxel-coords', className='info-value')
                ]),
                html.Div(className='info-row', children=[
                    html.Span("MNI Space Coord (mm)", className='info-label'),
                    html.Span(id='display-mni-coords', className='info-value')
                ]),
                html.Div(className='info-row', children=[
                    html.Span(id='stat-label-display', className='info-label', children="Voxel Value"),
                    html.Span(id='stat-value-display', className='info-value highlight-green')
                ]),
                html.Div(className='info-row', children=[
                    html.Span("Overlapping Studies (d ≤ 10mm)", className='info-label'),
                    html.Span(id='studies-count-display', className='info-value')
                ]),
                html.Div(className='info-row', children=[
                    html.Span("Bayesian Posterior Mean", className='info-label'),
                    html.Span(id='pmean-display', className='info-value highlight-purple')
                ]),
                
                html.Div(id='help-text-display', className='help-text')
            ])
        ]),
        
        # Right Side Plots and Contributing Studies
        html.Div(className='main-content', children=[
            # Brain Views Slices Row
            html.Div(className='dashboard-card', children=[
                html.H3(className='card-title', children=[
                    html.Span("🧠", className='card-title-icon secondary'),
                    "Orthogonal 3-Slice Viewer"
                ]),
                
                # 3 Brain Slices Grid
                html.Div(className='slices-grid', children=[
                    # Sagittal Slice
                    html.Div(className='slice-card', children=[
                        html.Div(className='slice-title-banner', children=[
                            html.Span("Sagittal View (Y-Z)", className='slice-title'),
                            html.Span(id='sagittal-slice-badge', className='slice-coordinate-badge')
                        ]),
                        dcc.Graph(
                            id='sagittal-graph',
                            config={'displayModeBar': False, 'scrollZoom': False},
                            style={'width': '100%', 'height': '270px'}
                        ),
                        html.Div(className='slider-container', style={'width': '95%'}, children=[
                            html.Div(className='slider-label-row', children=[
                                html.Span("Sagittal Slice (i)", className='control-label', style={'fontSize': '0.75rem'}),
                                html.Span(f"i={init_i}", id='sagittal-slider-val-badge', className='slider-value-badge')
                            ]),
                            dcc.Slider(
                                id='sagittal-slider', min=0, max=max_i, step=1, value=init_i,
                                marks=None, tooltip=None
                            )
                        ])
                    ]),
                    
                    # Coronal Slice
                    html.Div(className='slice-card', children=[
                        html.Div(className='slice-title-banner', children=[
                            html.Span("Coronal View (X-Z)", className='slice-title'),
                            html.Span(id='coronal-slice-badge', className='slice-coordinate-badge')
                        ]),
                        dcc.Graph(
                            id='coronal-graph',
                            config={'displayModeBar': False, 'scrollZoom': False},
                            style={'width': '100%', 'height': '270px'}
                        ),
                        html.Div(className='slider-container', style={'width': '95%'}, children=[
                            html.Div(className='slider-label-row', children=[
                                html.Span("Coronal Slice (j)", className='control-label', style={'fontSize': '0.75rem'}),
                                html.Span(f"j={init_j}", id='coronal-slider-val-badge', className='slider-value-badge')
                            ]),
                            dcc.Slider(
                                id='coronal-slider', min=0, max=max_j, step=1, value=init_j,
                                marks=None, tooltip=None
                            )
                        ])
                    ]),
                    
                    # Axial Slice
                    html.Div(className='slice-card', children=[
                        html.Div(className='slice-title-banner', children=[
                            html.Span("Axial View (X-Y)", className='slice-title'),
                            html.Span(id='axial-slice-badge', className='slice-coordinate-badge')
                        ]),
                        dcc.Graph(
                            id='axial-graph',
                            config={'displayModeBar': False, 'scrollZoom': False},
                            style={'width': '100%', 'height': '270px'}
                        ),
                        html.Div(className='slider-container', style={'width': '95%'}, children=[
                            html.Div(className='slider-label-row', children=[
                                html.Span("Axial Slice (k)", className='control-label', style={'fontSize': '0.75rem'}),
                                html.Span(f"k={init_k}", id='axial-slider-val-badge', className='slider-value-badge')
                            ]),
                            dcc.Slider(
                                id='axial-slider', min=0, max=max_k, step=1, value=init_k,
                                marks=None, tooltip=None
                            )
                        ])
                    ])
                ])
            ]),
            
            # Contributing Studies Table
            html.Div(className='dashboard-card', children=[
                html.H3(className='card-title', children=[
                    html.Span("📂", className='card-title-icon primary'),
                    "Contributing Functional Studies Lookup"
                ]),
                html.P("Listing studies reporting coordinate activations within 10mm (Euclidean distance) of the active MNI location:", style={'color': '#94a3b8', 'fontSize': '0.9rem', 'marginTop': '-0.5rem', 'marginBottom': '1rem'}),
                html.Div(className='table-container', children=[
                    html.Table(className='custom-table', children=[
                        html.Thead(html.Tr([
                            html.Th("PMID"),
                            html.Th("Sample Size (N)"),
                            html.Th("Modality"),
                            html.Th("Distance to Peak")
                        ])),
                        html.Tbody(id='studies-table-body')
                    ])
                ])
            ])
        ])
    ])
])

# Callback 1: State storage coordination
@app.callback(
    Output('active-coord-store', 'data'),
    [Input('axial-slider', 'value'),
     Input('coronal-slider', 'value'),
     Input('sagittal-slider', 'value'),
     Input('axial-graph', 'clickData'),
     Input('coronal-graph', 'clickData'),
     Input('sagittal-graph', 'clickData'),
     Input('coord-update-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks'),
     Input('input-x', 'n_submit'),
     Input('input-y', 'n_submit'),
     Input('input-z', 'n_submit'),
     Input('input-i', 'n_submit'),
     Input('input-j', 'n_submit'),
     Input('input-k', 'n_submit')],
    [State('active-coord-store', 'data'),
     State('input-x', 'value'),
     State('input-y', 'value'),
     State('input-z', 'value'),
     State('input-i', 'value'),
     State('input-j', 'value'),
     State('input-k', 'value')]
)
def update_store(axial_val, coronal_val, sagittal_val, axial_click, coronal_click, sagittal_click, update_clicks, reset_clicks, 
                 ns_x, ns_y, ns_z, ns_i, ns_j, ns_k, current_data, in_x, in_y, in_z, in_i, in_j, in_k):
    
    ctx = callback_context
    if not ctx.triggered:
        return current_data
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    new_data = current_data.copy()
    
    if trigger_id == 'reset-btn':
        reset_i, reset_j, reset_k = mni_to_voxel(0.0, 0.0, 18.0)
        new_data = {'i': reset_i, 'j': reset_j, 'k': reset_k}
    elif trigger_id == 'axial-slider':
        new_data['k'] = int(axial_val)
    elif trigger_id == 'coronal-slider':
        new_data['j'] = int(coronal_val)
    elif trigger_id == 'sagittal-slider':
        new_data['i'] = int(sagittal_val)
    elif trigger_id == 'axial-graph':
        if axial_click and 'points' in axial_click:
            new_data['i'] = int(axial_click['points'][0]['x'])
            new_data['j'] = int(axial_click['points'][0]['y'])
    elif trigger_id == 'coronal-graph':
        if coronal_click and 'points' in coronal_click:
            new_data['i'] = int(coronal_click['points'][0]['x'])
            new_data['k'] = int(coronal_click['points'][0]['y'])
    elif trigger_id == 'sagittal-graph':
        if sagittal_click and 'points' in sagittal_click:
            new_data['j'] = int(sagittal_click['points'][0]['x'])
            new_data['k'] = int(sagittal_click['points'][0]['y'])
    elif trigger_id in ['coord-update-btn', 'input-x', 'input-y', 'input-z', 'input-i', 'input-j', 'input-k']:
        # Voxel overrides take priority if provided
        if in_i is not None and in_j is not None and in_k is not None:
            new_data = {
                'i': int(np.clip(in_i, 0, brain_data.shape[0] - 1)),
                'j': int(np.clip(in_j, 0, brain_data.shape[1] - 1)),
                'k': int(np.clip(in_k, 0, brain_data.shape[2] - 1))
            }
        elif in_x is not None and in_y is not None and in_z is not None:
            vi, vj, vk = mni_to_voxel(in_x, in_y, in_z)
            new_data = {'i': vi, 'j': vj, 'k': vk}
            
    return new_data

# Callback 2: UI rendering & value matching
@app.callback(
    [Output('axial-graph', 'figure'),
     Output('coronal-graph', 'figure'),
     Output('sagittal-graph', 'figure'),
     Output('axial-slider', 'value'),
     Output('coronal-slider', 'value'),
     Output('sagittal-slider', 'value'),
     Output('axial-slider-val-badge', 'children'),
     Output('coronal-slider-val-badge', 'children'),
     Output('sagittal-slider-val-badge', 'children'),
     Output('axial-slice-badge', 'children'),
     Output('coronal-slice-badge', 'children'),
     Output('sagittal-slice-badge', 'children'),
     Output('display-voxel-coords', 'children'),
     Output('display-mni-coords', 'children'),
     Output('stat-value-display', 'children'),
     Output('stat-label-display', 'children'),
     Output('pmean-display', 'children'),
     Output('studies-count-display', 'children'),
     Output('studies-table-body', 'children'),
     Output('input-x', 'value'),
     Output('input-y', 'value'),
     Output('input-z', 'value'),
     Output('input-i', 'value'),
     Output('input-j', 'value'),
     Output('input-k', 'value'),
     Output('help-text-display', 'children')],
    [Input('active-coord-store', 'data'),
     Input('modality-dropdown', 'value'),
     Input('map-type-dropdown', 'value')]
)
def update_ui(coord_data, modality, map_type):
    i = int(coord_data.get('i', init_i))
    j = int(coord_data.get('j', init_j))
    k = int(coord_data.get('k', init_k))
    
    # Clip coordinates to safe ranges
    i = int(np.clip(i, 0, brain_data.shape[0] - 1))
    j = int(np.clip(j, 0, brain_data.shape[1] - 1))
    k = int(np.clip(k, 0, brain_data.shape[2] - 1))
    
    # 1. Convert to MNI coordinates
    x, y, z = voxel_to_mni(i, j, k)
    
    # 2. Get active map data
    active_map = maps[modality][map_type]
    stat_val = 0.0
    if active_map is not None:
        stat_val = float(active_map[i, j, k])
        
    # Get overall min and max of active map for normalization
    if active_map is not None and np.any(active_map > 0):
        stat_max = float(active_map.max())
        stat_min = float(active_map[active_map > 0].min())
    else:
        stat_max = 1.0
        stat_min = 0.0
        
    # Get active colormap
    colormap = COLORMAPS[map_type]
    
    # 3. Generate RGBA slices
    axial_rgba = get_axial_slice(brain_data, active_map, k, stat_min, stat_max, colormap)
    coronal_rgba = get_coronal_slice(brain_data, active_map, j, stat_min, stat_max, colormap)
    sagittal_rgba = get_sagittal_slice(brain_data, active_map, i, stat_min, stat_max, colormap)
    
    # 4. Create figures using Plotly
    # Axial figure
    fig_axial = px.imshow(axial_rgba, origin='lower')
    fig_axial.update_traces(hoverinfo='skip', hovertemplate=None)
    fig_axial.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, ticks=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, ticks=''),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        hovermode=False
    )
    fig_axial.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_axial.add_shape(type="line", x0=i, y0=0, x1=i, y1=brain_data.shape[1] - 1, line=dict(color="#8b5cf6", width=1.5, dash="dash"))
    fig_axial.add_shape(type="line", x0=0, y0=j, x1=brain_data.shape[0] - 1, y1=j, line=dict(color="#8b5cf6", width=1.5, dash="dash"))
    
    # Coronal figure
    fig_coronal = px.imshow(coronal_rgba, origin='lower')
    fig_coronal.update_traces(hoverinfo='skip', hovertemplate=None)
    fig_coronal.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, ticks=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, ticks=''),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        hovermode=False
    )
    fig_coronal.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_coronal.add_shape(type="line", x0=i, y0=0, x1=i, y1=brain_data.shape[2] - 1, line=dict(color="#8b5cf6", width=1.5, dash="dash"))
    fig_coronal.add_shape(type="line", x0=0, y0=k, x1=brain_data.shape[0] - 1, y1=k, line=dict(color="#8b5cf6", width=1.5, dash="dash"))
    
    # Sagittal figure
    fig_sagittal = px.imshow(sagittal_rgba, origin='lower')
    fig_sagittal.update_traces(hoverinfo='skip', hovertemplate=None)
    fig_sagittal.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, ticks=''),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, ticks=''),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        dragmode=False,
        hovermode=False
    )
    fig_sagittal.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_sagittal.add_shape(type="line", x0=j, y0=0, x1=j, y1=brain_data.shape[2] - 1, line=dict(color="#8b5cf6", width=1.5, dash="dash"))
    fig_sagittal.add_shape(type="line", x0=0, y0=k, x1=brain_data.shape[1] - 1, y1=k, line=dict(color="#8b5cf6", width=1.5, dash="dash"))
    
    # 5. Query coordinate stats
    k_count, post_mean, contributing_list = query_coordinate_stats(x, y, z, modality=modality)
    
    # Build studies table rows
    if contributing_list:
        table_rows = []
        for study in contributing_list:
            table_rows.append(html.Tr([
                html.Td(study['PMID']),
                html.Td(str(study['sample_size'])),
                html.Td(study['modality']),
                html.Td(f"{study['distance']} mm")
            ]))
    else:
        table_rows = [html.Tr([
            html.Td("No active studies within 10mm", colSpan=4, style={'textAlign': 'center', 'color': '#64748b'})
        ])]
        
    # Badges and UI text updates
    axial_badge = f"z = {round(z, 1)} mm"
    coronal_badge = f"y = {round(y, 1)} mm"
    sagittal_badge = f"x = {round(x, 1)} mm"
    
    axial_slider_label = f"k = {k}"
    coronal_slider_label = f"j = {j}"
    sagittal_slider_label = f"i = {i}"
    
    voxel_coords_text = f"I={i}, J={j}, K={k}"
    mni_coords_text = f"X={round(x, 1)}, Y={round(y, 1)}, Z={round(z, 1)}"
    
    map_label = "Exceedance Prob" if map_type == 'exceedance' else "Posterior Mean"
    stat_val_text = f"{stat_val:.5f}" if stat_val > 0 else "0.00000"
    pmean_val_text = f"{post_mean:.5f}"
    
    N_val = priors[modality]['N']
    studies_count_text = f"{k_count} / {N_val}"
    
    alpha_0 = priors[modality]['alpha_0']
    beta_0 = priors[modality]['beta_0']
    help_text_children = [
        f"Posterior Mean is updated dynamically: (α₀ + k) / (α₀ + β₀ + N) with N = {N_val}, and priors calculated dynamically from the {modality} mask pipeline: ",
        html.Code(f"α₀={alpha_0:.4f}, β₀={beta_0:.4f}")
    ]
    
    return (
        fig_axial, fig_coronal, fig_sagittal,
        k, j, i, # Sliders
        axial_slider_label, coronal_slider_label, sagittal_slider_label,
        axial_badge, coronal_badge, sagittal_badge,
        voxel_coords_text, mni_coords_text,
        stat_val_text, map_label, pmean_val_text, studies_count_text,
        table_rows,
        round(x, 1), round(y, 1), round(z, 1),
        i, j, k,
        help_text_children
    )

def main():
    print("Starting Dash application server...")
    app.run(debug=False, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()

