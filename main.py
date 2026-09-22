import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb

# ==============================================================================
# PREGUNTA 1: SATURACIÓN SELECTIVA (HS & L*c*h*)
# ==============================================================================

def interpolate_periodic_m(h_vals, control_points):
    """
    Interpolación lineal por tramos de m(h) con continuidad periódica en [0, 1].
    control_points: lista de tuplas (h_i, m_i) con h_i en [0, 1].
    """
    pts = sorted(control_points, key=lambda x: x[0])
    h_knots = np.array([p[0] for p in pts], dtype=float)
    m_knots = np.array([p[1] for p in pts], dtype=float)
    
    # Extensión periódica en las fronteras para evitar discontinuidades en el rojo
    h_ext = np.concatenate(([h_knots[-1] - 1.0], h_knots, [h_knots[0] + 1.0]))
    m_ext = np.concatenate(([m_knots[-1]], m_knots, [m_knots[0]]))
    
    h_mod = np.mod(h_vals, 1.0)
    return np.interp(h_mod, h_ext, m_ext)

def g_m(c, m):
    """
    Función de transformación de saturación/croma.
    Valor neutro: m = 1.0.
    m > 1: amplificación, m < 1: atenuación, m = 0: desaturación total.
    """
    return c * np.maximum(m, 0.0)

def color_saturation(img_rgb, control_points, mode='HS'):
    """
    Modifica selectivamente la saturación en función del tono.
    mode: 'HS' (HSV) o 'Lch' (CIE L*c*h*)
    """
    img_float = img_rgb.astype(np.float64) / 255.0 if img_rgb.max() > 1.0 else img_rgb.astype(np.float64)
    
    if mode == 'HS':
        hsv = rgb2hsv(img_float)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m_map = interpolate_periodic_m(H, control_points)
        S_mod = np.clip(g_m(S, m_map), 0.0, 1.0)
        hsv_mod = np.stack([H, S_mod, V], axis=-1)
        res_rgb = hsv2rgb(hsv_mod)
    elif mode == 'Lch':
        lab = rgb2lab(img_float)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        c_star = np.sqrt(a**2 + b**2)
        h_rad = np.arctan2(b, a)
        h_star = np.mod(h_rad / (2 * np.pi), 1.0)  # Normalizado en [0, 1]
        
        m_map = interpolate_periodic_m(h_star, control_points)
        c_star_mod = g_m(c_star, m_map)
        
        # Reconstrucción de canales a* y b*
        a_mod = c_star_mod * np.cos(h_rad)
        b_mod = c_star_mod * np.sin(h_rad)
        lab_mod = np.stack([L, a_mod, b_mod], axis=-1)
        res_rgb = lab2rgb(lab_mod)
    else:
        raise ValueError("Modo desconocido. Seleccione 'HS' o 'Lch'.")
        
    return np.clip(res_rgb, 0.0, 1.0)


# ==============================================================================
# PREGUNTA 2: ECUALIZACIÓN LOCAL Y CONTROL DE CONTRASTE
# ==============================================================================

def compute_region_cdf(region, num_bins=256, clip_limit=None):
    """
    Calcula la CDF normalizada con control de contraste mediante recorte (CLAHE).
    """
    hist, _ = np.histogram(region, bins=num_bins, range=(0.0, 1.0))
    if clip_limit is not None and clip_limit > 0:
        excess = np.maximum(hist - clip_limit, 0).sum()
        hist = np.minimum(hist, clip_limit) + excess / num_bins
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
    return cdf

def local_histogram_equalization(img_gray, region_size=(64, 64), step_size=(32, 32), num_bins=256, clip_limit=None):
    """
    Ecualización local sobre malla superpuesta con ventana de Hann para transiciones suaves.
    """
    H, W = img_gray.shape
    r_h, r_w = region_size
    s_h, s_w = step_size
    
    out_img = np.zeros_like(img_gray, dtype=np.float64)
    weight_sum = np.zeros_like(img_gray, dtype=np.float64)
    
    # Ventana de ponderación continua 2D
    window = np.outer(np.hanning(r_h), np.hanning(r_w))
    
    for y in range(0, max(1, H - r_h + s_h), s_h):
        for x in range(0, max(1, W - r_w + s_w), s_w):
            y_end = min(y + r_h, H)
            x_end = min(x + r_w, W)
            y_start = max(0, y_end - r_h)
            x_start = max(0, x_end - r_w)
            
            patch = img_gray[y_start:y_end, x_start:x_end]
            cdf = compute_region_cdf(patch, num_bins=num_bins, clip_limit=clip_limit)
            
            # Mapeo por niveles de intensidad
            idx = np.clip((patch * (num_bins - 1)).astype(int), 0, num_bins - 1)
            eq_patch = cdf[idx]
            
            w = window[:y_end-y_start, :x_end-x_start]
            out_img[y_start:y_end, x_start:x_end] += eq_patch * w
            weight_sum[y_start:y_end, x_start:x_end] += w
            
    return np.clip(out_img / (weight_sum + 1e-8), 0.0, 1.0)


# ==============================================================================
# PREGUNTA 3: REESCALADO E INTERPOLACIÓN (VECINO PRÓXIMO & BILINEAL)
# ==============================================================================

def resize_image(img, scale, mode='bilinear'):
    """
    Reescala una imagen monocromática o RGB por un factor s en [0.5, 2.0].
    """
    is_color = (img.ndim == 3)
    H, W = img.shape[:2]
    out_H = int(np.round(H * scale))
    out_W = int(np.round(W * scale))
    
    # Grilla de coordenadas discretas de salida
    y_out, x_out = np.indices((out_H, out_W))
    
    # Mapeo centrado hacia las coordenadas de entrada
    y_in = (y_out + 0.5) / scale - 0.5
    x_in = (x_out + 0.5) / scale - 0.5
    
    if mode == 'nearest':
        y_nearest = np.clip(np.round(y_in).astype(int), 0, H - 1)
        x_nearest = np.clip(np.round(x_in).astype(int), 0, W - 1)
        return img[y_nearest, x_nearest]
        
    elif mode == 'bilinear':
        x0 = np.floor(x_in).astype(int)
        y0 = np.floor(y_in).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        wa = ((x1 - x_in) * (y1 - y_in))
        wb = ((x_in - x0) * (y1 - y_in))
        wc = ((x1 - x_in) * (y_in - y0))
        wd = ((x_in - x0) * (y_in - y0))
        
        # Delimitación en fronteras
        x0_c = np.clip(x0, 0, W - 1)
        x1_c = np.clip(x1, 0, W - 1)
        y0_c = np.clip(y0, 0, H - 1)
        y1_c = np.clip(y1, 0, H - 1)
        
        if is_color:
            wa, wb = wa[..., None], wb[..., None]
            wc, wd = wc[..., None], wd[..., None]
            
        res = (wa * img[y0_c, x0_c] +
               wb * img[y0_c, x1_c] +
               wc * img[y1_c, x0_c] +
               wd * img[y1_c, x1_c])
        return res
    else:
        raise ValueError("Modo desconocido. Seleccione 'nearest' o 'bilinear'.")


# ==============================================================================
# BONUS: DÉBAYERING RGGB
# ==============================================================================

def simulate_bayer(img_rgb):
    """
    Simula el patrón de muestreo del sensor Bayer RGGB.
    """
    H, W, _ = img_rgb.shape
    bayer = np.zeros((H, W), dtype=img_rgb.dtype)
    bayer[0::2, 0::2] = img_rgb[0::2, 0::2, 0]  # R
    bayer[0::2, 1::2] = img_rgb[0::2, 1::2, 1]  # G1
    bayer[1::2, 0::2] = img_rgb[1::2, 0::2, 1]  # G2
    bayer[1::2, 1::2] = img_rgb[1::2, 1::2, 2]  # B
    return bayer

def debayer_superpixel(bayer):
    """
    Método Super-Píxel: reduce la resolución espacial a la mitad agrupando celdas 2x2.
    """
    H, W = bayer.shape
    out_H, out_W = H // 2, W // 2
    r = bayer[0:out_H*2:2, 0:out_W*2:2]
    g1 = bayer[0:out_H*2:2, 1:out_W*2:2]
    g2 = bayer[1:out_H*2:2, 0:out_W*2:2]
    b = bayer[1:out_H*2:2, 1:out_W*2:2]
    return np.stack([r, (g1 + g2) / 2.0, b], axis=-1)

def debayer_bilinear(bayer):
    """
    Desmosaizado bilineal a resolución completa (H x W).
    """
    H, W = bayer.shape
    rgb = np.zeros((H, W, 3), dtype=np.float64)
    padded = np.pad(bayer, 1, mode='reflect')
    
    # Máscaras booleanas del arreglo RGGB
    y, x = np.indices((H, W))
    is_r = (y % 2 == 0) & (x % 2 == 0)
    is_g1 = (y % 2 == 0) & (x % 2 == 1)
    is_g2 = (y % 2 == 1) & (x % 2 == 0)
    is_b = (y % 2 == 1) & (x % 2 == 1)
    
    # Reconstrucción del canal verde (G)
    p = padded
    rgb[..., 1] = np.where(is_r | is_b, 
                           (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0, 
                           bayer)
    # Reconstrucción del canal rojo (R)
    r_val = np.zeros((H, W))
    r_val[is_r] = bayer[is_r]
    r_val[is_g1] = (p[1:-1, :-2][is_g1] + p[1:-1, 2:][is_g1]) / 2.0
    r_val[is_g2] = (p[:-2, 1:-1][is_g2] + p[2:, 1:-1][is_g2]) / 2.0
    r_val[is_b] = (p[:-2, :-2][is_b] + p[:-2, 2:][is_b] + p[2:, :-2][is_b] + p[2:, 2:][is_b]) / 4.0
    rgb[..., 0] = r_val
    
    # Reconstrucción del canal azul (B)
    b_val = np.zeros((H, W))
    b_val[is_b] = bayer[is_b]
    b_val[is_g1] = (p[:-2, 1:-1][is_g1] + p[2:, 1:-1][is_g1]) / 2.0
    b_val[is_g2] = (p[1:-1, :-2][is_g2] + p[1:-1, 2:][is_g2]) / 2.0
    b_val[is_r] = (p[:-2, :-2][is_r] + p[:-2, 2:][is_r] + p[2:, :-2][is_r] + p[2:, 2:][is_r]) / 4.0
    rgb[..., 2] = b_val
    
    return np.clip(rgb, 0.0, 1.0)


# ==============================================================================
# PIPELINE DE PRUEBA Y VALIDACIÓN
# ==============================================================================
if __name__ == '__main__':
    print("Iniciando validación de funciones implementadas...")
    
    # Generación de imagen sintética de prueba
    x = np.linspace(-3, 3, 256)
    xx, yy = np.meshgrid(x, x)
    synth_gray = (np.sin(xx**2 + yy**2) + 1.0) / 2.0
    synth_rgb = plt.cm.jet(synth_gray)[..., :3]

    # 1. Prueba Saturación Selectiva
    ctrl_pts = [(0.0, 1.0), (0.33, 2.5), (0.66, 0.2), (1.0, 1.0)]
    q1_hs = color_saturation(synth_rgb, ctrl_pts, mode='HS')
    q1_lch = color_saturation(synth_rgb, ctrl_pts, mode='Lch')
    
    # 2. Prueba Ecualización Local de Histograma
    q2_local = local_histogram_equalization(synth_gray, region_size=(32, 32), step_size=(16, 16))
    q2_clahe = local_histogram_equalization(synth_gray, region_size=(32, 32), step_size=(16, 16), clip_limit=15)
    
    # 3. Prueba Reescalado
    q3_near = resize_image(synth_rgb, scale=1.4, mode='nearest')
    q3_bilin = resize_image(synth_rgb, scale=1.4, mode='bilinear')
    
    # 4. Prueba Desmosaizado Bayer
    bayer_mat = simulate_bayer(synth_rgb)
    debayer_res = debayer_bilinear(bayer_mat)
    
    print("Ejecución finalizada con éxito. Todos los módulos operativos.")
