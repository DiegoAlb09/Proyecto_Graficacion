import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.integrate import ode
from scipy.ndimage import zoom
import colorsys
import os

class EulerArtGenerator:
    """
    Generador de Arte Procedural basado en las Ecuaciones de Euler en 2D
    VERSIÓN INTERACTIVA - El usuario puede personalizar todos los parámetros
    """
    
    def __init__(self, width=800, height=600, resolution=200):
        """
        Inicializa el generador de arte
        
        Args:
            width (int): Ancho del canvas en píxeles
            height (int): Alto del canvas en píxeles  
            resolution (int): Resolución de la grilla computacional
        """
        self.width = width
        self.height = height
        
        # Ajustar resolución si es necesaria
        max_safe_resolution = min(width, height)
        if resolution > max_safe_resolution:
            print(f"⚠️  Resolución {resolution} muy alta para dimensiones {width}x{height}.")
            print(f"Ajustando resolución a {max_safe_resolution}.")
            resolution = max_safe_resolution
        
        self.resolution = resolution
        
        # Crear grilla espacial para los cálculos
        self.x = np.linspace(0, width, resolution)
        self.y = np.linspace(0, height, resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Parámetros físicos (se pueden personalizar)
        self.viscosity = 0.005
        self.dt = 0.005
        self.time = 0
        
        # Campos de velocidad (componentes u, v)
        self.u = np.zeros_like(self.X)
        self.v = np.zeros_like(self.Y)
        
        # Campo de vorticidad (rotacional de la velocidad)
        self.vorticity = np.zeros_like(self.X)
        
        # Canvas para el arte final
        self.canvas = np.zeros((height, width, 3))
        
        # Variables para fuerzas externas dinámicas
        self.force_centers = []
        self.force_strengths = []
        self.force_evolution_rate = 0.1
        
    def set_physical_parameters(self, viscosity=0.005, dt=0.005, force_rate=0.1):
        """
        Permite al usuario personalizar los parámetros físicos de la simulación
        """
        self.viscosity = viscosity
        self.dt = dt
        self.force_evolution_rate = force_rate
        print(f"✅ Parámetros físicos actualizados:")
        print(f"   - Viscosidad: {viscosity}")
        print(f"   - Paso temporal: {dt}")
        print(f"   - Tasa de evolución de fuerzas: {force_rate}")
        
    def initialize_flow_field(self, flow_type='vortex_dance', **kwargs):
        """
        Inicializa el campo de flujo con diferentes patrones
        El usuario puede personalizar los parámetros de cada tipo de flujo
        """
        if flow_type == 'vortex_dance':
            # Parámetros personalizables para vórtices
            n_vortices = kwargs.get('n_vortices', 4)
            strength_range = kwargs.get('strength_range', (3.0, 5.0))
            positions = kwargs.get('positions', 'corners')  # 'corners', 'random', 'center', 'custom'
            custom_positions = kwargs.get('custom_positions', [])  # Para posiciones personalizadas
            
            if positions == 'corners' and n_vortices == 4:
                centers = [(0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)]
            elif positions == 'center':
                centers = [(0.5, 0.5)]
            elif positions == 'random':
                np.random.seed(kwargs.get('seed', 42))
                centers = [(np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)) 
                          for _ in range(n_vortices)]
            elif positions == 'custom' and custom_positions:
                centers = custom_positions[:n_vortices]
            else:
                centers = [(0.5, 0.5)]  # Por defecto
            
            # Generar fuerzas
            np.random.seed(kwargs.get('seed', 42))
            strengths = [np.random.uniform(strength_range[0], strength_range[1]) * 
                        (1 if i % 2 == 0 else -1) for i in range(len(centers))]
            
            self.force_centers = [(cx * self.width, cy * self.height) for cx, cy in centers]
            self.force_strengths = strengths.copy()
            
            decay_factor = kwargs.get('decay_factor', 0.15)
            core_size = kwargs.get('core_size', 10)
            
            for (cx, cy), strength in zip(centers, strengths):
                cx_abs, cy_abs = cx * self.width, cy * self.height
                
                dx = self.X - cx_abs
                dy = self.Y - cy_abs
                r_squared = dx**2 + dy**2 + core_size
                
                decay = np.exp(-r_squared / (decay_factor * min(self.width, self.height)**2))
                
                self.u += -strength * dy / r_squared * decay
                self.v += strength * dx / r_squared * decay
                
        elif flow_type == 'spiral_galaxy':
            # Parámetros personalizables para espiral
            spiral_arms = kwargs.get('spiral_arms', 8)
            radial_strength = kwargs.get('radial_strength', 0.8)
            tangential_strength = kwargs.get('tangential_strength', 6.0)
            decay_rate = kwargs.get('decay_rate', 0.3)
            
            center_x, center_y = self.width/2, self.height/2
            dx = self.X - center_x
            dy = self.Y - center_y
            r = np.sqrt(dx**2 + dy**2) + 1e-6
            theta = np.arctan2(dy, dx)
            
            v_r = radial_strength * r * np.sin(spiral_arms * theta) * \
                  np.exp(-r / (decay_rate * min(self.width, self.height)))
            v_theta = tangential_strength / (1 + r/50) * \
                     np.exp(-r / (0.4 * min(self.width, self.height)))
            
            self.u = v_r * np.cos(theta) - v_theta * np.sin(theta)
            self.v = v_r * np.sin(theta) + v_theta * np.cos(theta)
            
        elif flow_type == 'turbulent_ocean':
            # Parámetros personalizables para turbulencia
            n_modes = kwargs.get('n_modes', 8)
            frequency_range = kwargs.get('frequency_range', (-0.08, 0.08))
            amplitude_range = kwargs.get('amplitude_range', (2.0, 4.0))
            seed = kwargs.get('seed', 42)
            
            np.random.seed(seed)
            
            for i in range(n_modes):
                kx = np.random.uniform(*frequency_range)
                ky = np.random.uniform(*frequency_range)
                amplitude = np.random.uniform(*amplitude_range)
                phase = np.random.uniform(0, 2*np.pi)
                
                wave = amplitude * np.sin(kx * self.X + ky * self.Y + phase)
                self.u += np.gradient(wave, axis=1) * 0.5
                self.v += np.gradient(wave, axis=0) * 0.5
                
        # Calcular vorticidad inicial
        self.calculate_vorticity()
        print(f"✅ Campo de flujo '{flow_type}' inicializado con parámetros personalizados")
    
    def calculate_vorticity(self):
        """Calcula la vorticidad del campo de velocidad"""
        dx = self.width / self.resolution
        dy = self.height / self.resolution
        
        du_dy = np.gradient(self.u, dy, axis=0)
        dv_dx = np.gradient(self.v, dx, axis=1)
        self.vorticity = np.clip(dv_dx - du_dy, -500, 500)
    
    def advect_field(self, field, u, v, dt):
        """Advecta un campo escalar usando método Runge-Kutta"""
        u_safe = np.clip(u, -50, 50)
        v_safe = np.clip(v, -50, 50)
        field_safe = np.nan_to_num(field, nan=0.0, posinf=50.0, neginf=-50.0)
        
        dx = self.width / self.resolution
        dy = self.height / self.resolution
        
        dudx = np.gradient(field_safe, dx, axis=1)
        dudy = np.gradient(field_safe, dy, axis=0)
        
        dudx = np.clip(dudx, -50, 50)
        dudy = np.clip(dudy, -50, 50)
        
        k1_u = -u_safe * dudx
        k1_v = -v_safe * dudy
        
        field_mid = field_safe + 0.5 * dt * (k1_u + k1_v)
        field_mid = np.nan_to_num(field_mid, nan=0.0)
        
        dudx_mid = np.gradient(field_mid, dx, axis=1)
        dudy_mid = np.gradient(field_mid, dy, axis=0)
        
        k2_u = -u_safe * np.clip(dudx_mid, -50, 50)
        k2_v = -v_safe * np.clip(dudy_mid, -50, 50)
        
        result = field_safe + dt * (k2_u + k2_v)
        
        return np.clip(np.nan_to_num(result, nan=0.0), -500, 500)
    
    def apply_viscosity(self, field, viscosity, dt):
        """Aplica difusión viscosa"""
        field_safe = np.nan_to_num(field, nan=0.0, posinf=50.0, neginf=-50.0)
        
        dx = self.width / self.resolution
        dy = self.height / self.resolution
        
        d2_dx2 = (np.roll(field_safe, 1, axis=1) - 2*field_safe + np.roll(field_safe, -1, axis=1)) / (dx**2)
        d2_dy2 = (np.roll(field_safe, 1, axis=0) - 2*field_safe + np.roll(field_safe, -1, axis=0)) / (dy**2)
        
        laplacian = d2_dx2 + d2_dy2
        laplacian = np.clip(laplacian, -100, 100)
        
        result = field_safe + viscosity * dt * laplacian
        
        return np.clip(np.nan_to_num(result, nan=0.0), -500, 500)
    
    def add_dynamic_forces(self, force_strength=3.0, n_forces=2):
        """Añade fuerzas externas dinámicas personalizables"""
        t = self.time * self.force_evolution_rate
        
        for i in range(n_forces):
            angle = t + i * np.pi
            radius = 0.15 * min(self.width, self.height)
            
            center_x = self.width/2 + radius * np.cos(angle)
            center_y = self.height/2 + radius * np.sin(angle)
            strength = force_strength * np.cos(t + i)
            
            dx = self.X - center_x
            dy = self.Y - center_y
            r_squared = dx**2 + dy**2 + 25
            
            decay = np.exp(-r_squared / (0.1 * min(self.width, self.height)**2))
            
            self.u += -strength * dy / r_squared * decay * 0.05
            self.v += strength * dx / r_squared * decay * 0.05
    
    def step_simulation(self, noise_strength=0.3, energy_injection=True):
        """Avanza la simulación con parámetros personalizables"""
        self.u = np.nan_to_num(self.u, nan=0.0)
        self.v = np.nan_to_num(self.v, nan=0.0)
        self.u = np.clip(self.u, -50, 50)
        self.v = np.clip(self.v, -50, 50)
        
        # Añadir fuerzas dinámicas
        self.add_dynamic_forces()
        
        # Advección y difusión
        self.vorticity = self.advect_field(self.vorticity, self.u, self.v, self.dt)
        self.vorticity = self.apply_viscosity(self.vorticity, self.viscosity, self.dt)
        
        self.u = self.advect_field(self.u, self.u, self.v, self.dt)
        self.v = self.advect_field(self.v, self.u, self.v, self.dt)
        
        self.u = self.apply_viscosity(self.u, self.viscosity, self.dt)
        self.v = self.apply_viscosity(self.v, self.viscosity, self.dt)
        
        # Ruido aleatorio opcional
        if self.time % 10 == 0 and noise_strength > 0:
            self.u += noise_strength * np.random.randn(*self.u.shape) * 0.02
            self.v += noise_strength * np.random.randn(*self.v.shape) * 0.02
        
        # Inyección de energía opcional
        if energy_injection and self.time % 20 == 0:
            center_x, center_y = self.width//2, self.height//2
            
            grid_spacing_x = self.width / self.resolution
            grid_spacing_y = self.height / self.resolution
            
            cx = int(np.clip(center_x / grid_spacing_x, 2, self.resolution - 3))
            cy = int(np.clip(center_y / grid_spacing_y, 2, self.resolution - 3))
            
            self.u[cy-1:cy+2, cx-1:cx+2] += np.random.randn(3, 3) * 0.5
            self.v[cy-1:cy+2, cx-1:cx+2] += np.random.randn(3, 3) * 0.5
        
        # Condiciones de frontera
        self.u[0, :] = self.u[-1, :] = 0
        self.u[:, 0] = self.u[:, -1] = 0
        self.v[0, :] = self.v[-1, :] = 0
        self.v[:, 0] = self.v[:, -1] = 0
        
        self.calculate_vorticity()
        self.time += 1
    
    def field_to_color(self, field, colormap='hsv', normalize=True, enhance_contrast=True, 
                      gamma=0.8, saturation=0.98):
        """Convierte un campo escalar a colores RGB con parámetros personalizables"""
        field_safe = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
        
        if normalize:
            field_min = np.percentile(field_safe, 5)
            field_max = np.percentile(field_safe, 95)
            
            if np.abs(field_max - field_min) < 1e-6:
                field_min = 0.0
                field_max = 1.0
            
            field_norm = np.clip((field_safe - field_min) / (field_max - field_min + 1e-8), 0, 1)
            
            if enhance_contrast:
                field_norm = np.power(field_norm, gamma)
        else:
            field_norm = np.clip(field_safe, 0, 1)
        
        if colormap == 'hsv':
            h = field_norm
            s = np.ones_like(h) * saturation
            v = 0.3 + 0.7 * field_norm
            
            rgb = np.zeros((*field.shape, 3))
            for i in range(field.shape[0]):
                for j in range(field.shape[1]):
                    rgb[i, j] = colorsys.hsv_to_rgb(h[i, j], s[i, j], v[i, j])
            return rgb
            
        elif colormap == 'plasma':
            cmap = plt.cm.plasma
            return cmap(field_norm)[:, :, :3]
            
        elif colormap == 'viridis':
            cmap = plt.cm.viridis
            return cmap(field_norm)[:, :, :3]
            
        elif colormap == 'inferno':
            cmap = plt.cm.inferno
            return cmap(field_norm)[:, :, :3]
            
        elif colormap == 'magma':
            cmap = plt.cm.magma
            return cmap(field_norm)[:, :, :3]
    
    def velocity_to_color(self, enhance_contrast=True, gamma=0.7):
        """Convierte el campo de velocidad a colores"""
        u_safe = np.nan_to_num(self.u, nan=0.0)
        v_safe = np.nan_to_num(self.v, nan=0.0)
        
        velocity_mag = np.sqrt(u_safe**2 + v_safe**2)
        velocity_angle = np.arctan2(v_safe, u_safe)
        
        hue = (velocity_angle + np.pi) / (2 * np.pi)
        
        max_vel = np.percentile(velocity_mag, 90)
        if max_vel < 1e-6:
            max_vel = 1.0
            
        saturation = np.clip(velocity_mag / (max_vel + 1e-8), 0, 1)
        
        if enhance_contrast:
            saturation = np.power(saturation, gamma)
        
        value = 0.4 + 0.6 * saturation
        
        rgb = np.zeros((*self.X.shape, 3))
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                h = hue[i, j]
                s = saturation[i, j]
                v = value[i, j]
                
                if np.isnan(h) or np.isnan(s) or np.isnan(v) or h < 0 or h > 1 or s < 0 or s > 1 or v < 0 or v > 1:
                    h = 0.0
                    s = 0.0
                    v = 0.5
                    
                rgb[i, j] = colorsys.hsv_to_rgb(h, s, v)
        
        return rgb
    
    def create_art_frame(self, art_style='vorticity_flow', mix_alpha=0.6, colormap='hsv'):
        """Genera un frame de arte con estilo personalizable"""
        if art_style == 'vorticity_flow':
            colors = self.field_to_color(self.vorticity, colormap, enhance_contrast=True)
            
        elif art_style == 'velocity_field':
            colors = self.velocity_to_color(enhance_contrast=True)
            
        elif art_style == 'mixed_media':
            vort_colors = self.field_to_color(self.vorticity, 'plasma', enhance_contrast=True)
            vel_colors = self.velocity_to_color(enhance_contrast=True)
            
            vort_colors = np.nan_to_num(vort_colors, nan=0.0)
            vel_colors = np.nan_to_num(vel_colors, nan=0.0)
            
            alpha = mix_alpha + 0.2 * np.sin(self.time * 0.05)
            colors = alpha * vort_colors + (1 - alpha) * vel_colors
            colors = np.nan_to_num(colors, nan=0.0)
        
        # Redimensionar con interpolación de calidad
        zoom_factor_y = self.height / colors.shape[0]
        zoom_factor_x = self.width / colors.shape[1]
        
        if zoom_factor_x > 1 or zoom_factor_y > 1:
            colors_resized = zoom(colors, (zoom_factor_y, zoom_factor_x, 1), order=3)
        else:
            colors_resized = zoom(colors, (zoom_factor_y, zoom_factor_x, 1), order=1)
        
        return np.clip(colors_resized, 0, 1)
    
    def generate_static_art(self, steps=300, flow_type='vortex_dance', 
                           art_style='mixed_media', save_path=None, dpi=600,
                           flow_params=None, style_params=None):
        """
        Genera una obra de arte estática con parámetros completamente personalizables
        """
        print(f"🎨 Inicializando flujo tipo: {flow_type}")
        
        # Usar parámetros personalizados si se proporcionan
        if flow_params is None:
            flow_params = {}
        
        self.initialize_flow_field(flow_type, **flow_params)
        
        print(f"⚙️  Simulando {steps} pasos temporales...")
        for i in range(steps):
            if i % 50 == 0:
                print(f"   Paso {i}/{steps}")
            self.step_simulation()
        
        print(f"🖌️  Generando arte con estilo: {art_style}")
        
        # Usar parámetros de estilo personalizados si se proporcionan
        if style_params is None:
            style_params = {}
        
        final_art = self.create_art_frame(art_style, **style_params)
        
        if save_path:
            plt.figure(figsize=(12, 9))
            plt.imshow(final_art, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Arte Procedural HD - Ecuaciones de Euler 2D\n'
                     f'Flujo: {flow_type}, Estilo: {art_style}', 
                     fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            plt.show()
            print(f"✅ Arte HD guardado en: {save_path}")
        else:
            plt.figure(figsize=(10, 8))
            plt.imshow(final_art, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Arte Procedural - {flow_type} / {art_style}', fontsize=12)
            plt.show()
        
        return final_art
    
    def create_animation(self, frames=100, flow_type='vortex_dance', 
                        art_style='mixed_media', save_path=None,
                        flow_params=None, style_params=None):
        """Crea una animación con parámetros personalizables"""
        print(f"🎬 Creando animación HD con {frames} frames...")
        
        if flow_params is None:
            flow_params = {}
        if style_params is None:
            style_params = {}
        
        self.initialize_flow_field(flow_type, **flow_params)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.axis('off')
        fig.patch.set_facecolor('black')
        fig.tight_layout()
        
        art_frame = self.create_art_frame(art_style, **style_params)
        im = ax.imshow(art_frame, extent=[0, self.width, 0, self.height], 
                      interpolation='bilinear')
        
        def animate(frame):
            for _ in range(3):
                self.step_simulation()
            
            art_frame = self.create_art_frame(art_style, **style_params)
            im.set_array(art_frame)
            
            ax.set_title(f'Arte Procedural HD - Frame {frame}\n'
                        f'{flow_type} / {art_style} (Tiempo: {self.time})', 
                        fontsize=12, color='white')
            
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=frames, 
                           interval=100, blit=True, repeat=True)
        
        if save_path:
            print(f"💾 Guardando animación HD en: {save_path}")
            anim.save(save_path, writer='pillow', fps=10, 
                     savefig_kwargs={'facecolor': 'black'})
        
        plt.show()
        return anim


def get_user_input():
    """
    Función para obtener parámetros del usuario de forma interactiva
    """
    print("=" * 60)
    print("🎨 GENERADOR DE ARTE PROCEDURAL INTERACTIVO")
    print("   Basado en las Ecuaciones de Euler 2D")
    print("=" * 60)
    
    # Configuración básica
    print("\n📐 CONFIGURACIÓN BÁSICA:")
    try:
        width = int(input("Ancho de la imagen (píxeles) [800]: ") or "800")
        height = int(input("Alto de la imagen (píxeles) [600]: ") or "600")
        resolution = int(input("Resolución de cálculo [200]: ") or "200")
    except ValueError:
        print("❌ Valores inválidos, usando valores por defecto")
        width, height, resolution = 800, 600, 200
    
    # Tipo de flujo
    print("\n🌊 TIPO DE FLUJO:")
    print("1. vortex_dance    - Danza de vórtices (recomendado)")
    print("2. spiral_galaxy   - Galaxia espiral")
    print("3. turbulent_ocean - Océano turbulento")
    
    flow_choice = input("Selecciona tipo de flujo [1]: ") or "1"
    flow_types = {"1": "vortex_dance", "2": "spiral_galaxy", "3": "turbulent_ocean"}
    flow_type = flow_types.get(flow_choice, "vortex_dance")
    
    # Estilo de arte
    print("\n🎨 ESTILO DE ARTE:")
    print("1. mixed_media     - Mezcla artística (recomendado)")
    print("2. vorticity_flow  - Campo de vorticidad")
    print("3. velocity_field  - Campo de velocidad")
    
    style_choice = input("Selecciona estilo de arte [1]: ") or "1"
    style_types = {"1": "mixed_media", "2": "vorticity_flow", "3": "velocity_field"}
    art_style = style_types.get(style_choice, "mixed_media")
    
    # Parámetros de simulación
    print("\n⚙️  PARÁMETROS DE SIMULACIÓN:")
    try:
        steps = int(input("Número de pasos de simulación [300]: ") or "300")
        viscosity = float(input("Viscosidad (0.001-0.01) [0.005]: ") or "0.005")
        dt = float(input("Paso temporal (0.001-0.01) [0.005]: ") or "0.005")
    except ValueError:
        print("❌ Valores inválidos, usando valores por defecto")
        steps, viscosity, dt = 300, 0.005, 0.005
    
    # Archivo de salida
    print("\n💾 ARCHIVO DE SALIDA:")
    filename = input("Nombre del archivo (sin extensión) [mi_arte_euler]: ") or "mi_arte_euler"
    save_path = f"{filename}.png"
    
    # Calidad
    print("\n🖼️  CALIDAD DE IMAGEN:")
    try:
        dpi = int(input("DPI (300-1200) [600]: ") or "600")
    except ValueError:
        dpi = 600
    
    # Tipo de output
    print("\n📱 TIPO DE SALIDA:")
    print("1. Imagen estática")
    print("2. Animación GIF")
    
    output_choice = input("Selecciona tipo de salida [1]: ") or "1"
    create_animation = output_choice == "2"
    
    if create_animation:
        try:
            frames = int(input("Número de frames para la animación [60]: ") or "60")
        except ValueError:
            frames = 60
        save_path = f"{filename}.gif"
    else:
        frames = None
    
    return {
        'width': width,
        'height': height,
        'resolution': resolution,
        'flow_type': flow_type,
        'art_style': art_style,
        'steps': steps,
        'viscosity': viscosity,
        'dt': dt,
        'save_path': save_path,
        'dpi': dpi,
        'create_animation': create_animation,
        'frames': frames
    }


def main():
    """Función principal interactiva"""
    
    # Obtener parámetros del usuario
    params = get_user_input()
    
    print("\n" + "=" * 60)
    print("🚀 INICIANDO GENERACIÓN DE ARTE...")
    print("=" * 60)
    
    # Crear el generador con los parámetros del usuario
    generator = EulerArtGenerator(
        width=params['width'], 
        height=params['height'], 
        resolution=params['resolution']
    )
    
    # Configurar parámetros físicos
    generator.set_physical_parameters(
        viscosity=params['viscosity'],
        dt=params['dt']
    )
    
    # Parámetros específicos según el tipo de flujo
    flow_params = {}
    
    if params['flow_type'] == 'vortex_dance':
        print("\n🌀 Configurando parámetros para Danza de Vórtices...")
        print("¿Deseas personalizar los vórtices? (s/n) [n]: ", end="")
        customize = input().lower() == 's'
        
        if customize:
            try:
                n_vortices = int(input("Número de vórtices [4]: ") or "4")
                strength_min = float(input("Fuerza mínima de vórtices [3.0]: ") or "3.0")
                strength_max = float(input("Fuerza máxima de vórtices [5.0]: ") or "5.0")
                
                print("Posiciones de vórtices:")
                print("1. Esquinas (corners)")
                print("2. Centro (center)")
                print("3. Aleatorio (random)")
                pos_choice = input("Selecciona [1]: ") or "1"
                positions = {"1": "corners", "2": "center", "3": "random"}
                position_type = positions.get(pos_choice, "corners")
                
                flow_params = {
                    'n_vortices': n_vortices,
                    'strength_range': (strength_min, strength_max),
                    'positions': position_type,
                    'seed': 42
                }
            except ValueError:
                print("❌ Valores inválidos, usando configuración por defecto")
    
    elif params['flow_type'] == 'spiral_galaxy':
        print("\n🌌 Configurando parámetros para Galaxia Espiral...")
        print("¿Deseas personalizar la espiral? (s/n) [n]: ", end="")
        customize = input().lower() == 's'
        
        if customize:
            try:
                spiral_arms = int(input("Número de brazos espirales [8]: ") or "8")
                radial_strength = float(input("Fuerza radial [0.8]: ") or "0.8")
                tangential_strength = float(input("Fuerza tangencial [6.0]: ") or "6.0")
                
                flow_params = {
                    'spiral_arms': spiral_arms,
                    'radial_strength': radial_strength,
                    'tangential_strength': tangential_strength
                }
            except ValueError:
                print("❌ Valores inválidos, usando configuración por defecto")
    
    elif params['flow_type'] == 'turbulent_ocean':
        print("\n🌊 Configurando parámetros para Océano Turbulento...")
        print("¿Deseas personalizar la turbulencia? (s/n) [n]: ", end="")
        customize = input().lower() == 's'
        
        if customize:
            try:
                n_modes = int(input("Número de modos de turbulencia [8]: ") or "8")
                freq_range = float(input("Rango de frecuencias [0.08]: ") or "0.08")
                amp_min = float(input("Amplitud mínima [2.0]: ") or "2.0")
                amp_max = float(input("Amplitud máxima [4.0]: ") or "4.0")
                
                flow_params = {
                    'n_modes': n_modes,
                    'frequency_range': (-freq_range, freq_range),
                    'amplitude_range': (amp_min, amp_max),
                    'seed': 42
                }
            except ValueError:
                print("❌ Valores inválidos, usando configuración por defecto")
    
    # Parámetros de estilo
    style_params = {}
    
    if params['art_style'] == 'mixed_media':
        print(f"\n🎨 Configurando estilo: {params['art_style']}...")
        print("¿Deseas personalizar la mezcla? (s/n) [n]: ", end="")
        customize = input().lower() == 's'
        
        if customize:
            try:
                mix_alpha = float(input("Factor de mezcla (0.0-1.0) [0.6]: ") or "0.6")
                style_params['mix_alpha'] = mix_alpha
            except ValueError:
                print("❌ Valor inválido, usando configuración por defecto")
    
    elif params['art_style'] == 'vorticity_flow':
        print(f"\n🌈 Configurando mapa de colores para {params['art_style']}...")
        print("1. HSV (multicolor)")
        print("2. Plasma (morado-rosa)")
        print("3. Viridis (verde-azul)")
        print("4. Inferno (negro-rojo-amarillo)")
        print("5. Magma (negro-morado-blanco)")
        
        cmap_choice = input("Selecciona mapa de colores [1]: ") or "1"
        cmaps = {"1": "hsv", "2": "plasma", "3": "viridis", "4": "inferno", "5": "magma"}
        colormap = cmaps.get(cmap_choice, "hsv")
        style_params['colormap'] = colormap
    
    # Confirmación final
    print(f"\n📋 RESUMEN DE CONFIGURACIÓN:")
    print(f"   🖼️  Dimensiones: {params['width']}x{params['height']} píxeles")
    print(f"   ⚙️  Resolución: {params['resolution']}")
    print(f"   🌊 Tipo de flujo: {params['flow_type']}")
    print(f"   🎨 Estilo de arte: {params['art_style']}")
    print(f"   🔢 Pasos de simulación: {params['steps']}")
    print(f"   💾 Archivo de salida: {params['save_path']}")
    
    if params['create_animation']:
        print(f"   🎬 Animación: {params['frames']} frames")
    else:
        print(f"   🖼️  DPI: {params['dpi']}")
    
    print(f"\n¿Proceder con la generación? (s/n) [s]: ", end="")
    confirm = input().lower()
    
    if confirm == 'n':
        print("❌ Generación cancelada por el usuario")
        return
    
    try:
        print(f"\n🎨 Iniciando generación...")
        
        if params['create_animation']:
            # Crear animación
            animation = generator.create_animation(
                frames=params['frames'],
                flow_type=params['flow_type'],
                art_style=params['art_style'],
                save_path=params['save_path'],
                flow_params=flow_params,
                style_params=style_params
            )
            print(f"✅ ¡Animación creada exitosamente!")
            
        else:
            # Crear imagen estática
            final_art = generator.generate_static_art(
                steps=params['steps'],
                flow_type=params['flow_type'],
                art_style=params['art_style'],
                save_path=params['save_path'],
                dpi=params['dpi'],
                flow_params=flow_params,
                style_params=style_params
            )
            print(f"✅ ¡Imagen creada exitosamente!")
        
        print(f"\n🎉 GENERACIÓN COMPLETADA")
        print(f"   📁 Archivo guardado: {params['save_path']}")
        print(f"   🎨 Tipo: {params['flow_type']} / {params['art_style']}")
        
    except Exception as e:
        print(f"❌ Error durante la generación: {str(e)}")
        print("💡 Intenta con parámetros diferentes o revisa la configuración")


def create_preset_examples():
    """
    Función para crear ejemplos predefinidos sin interacción del usuario
    """
    print("🎨 CREANDO EJEMPLOS PREDEFINIDOS...")
    
    # Ejemplo 1: Vórtices clásicos
    print("\n1️⃣  Creando: Danza de Vórtices Clásica")
    generator1 = EulerArtGenerator(width=600, height=600, resolution=200)
    art1 = generator1.generate_static_art(
        steps=200,
        flow_type='vortex_dance',
        art_style='mixed_media',
        save_path='ejemplo_vortices.png',
        dpi=400
    )
    
    # Ejemplo 2: Espiral galáctica
    print("\n2️⃣  Creando: Galaxia Espiral")
    generator2 = EulerArtGenerator(width=600, height=600, resolution=180)
    art2 = generator2.generate_static_art(
        steps=250,
        flow_type='spiral_galaxy',
        art_style='velocity_field',
        save_path='ejemplo_espiral.png',
        dpi=400,
        flow_params={'spiral_arms': 6, 'tangential_strength': 8.0}
    )
    
    # Ejemplo 3: Océano turbulento
    print("\n3️⃣  Creando: Océano Turbulento")
    generator3 = EulerArtGenerator(width=600, height=600, resolution=150)
    generator3.set_physical_parameters(viscosity=0.003, dt=0.004)
    art3 = generator3.generate_static_art(
        steps=180,
        flow_type='turbulent_ocean',
        art_style='vorticity_flow',
        save_path='ejemplo_oceano.png',
        dpi=400,
        flow_params={'n_modes': 12, 'amplitude_range': (1.5, 3.5)},
        style_params={'colormap': 'plasma'}
    )
    
    print("\n✅ ¡Todos los ejemplos creados exitosamente!")


def show_help():
    """
    Muestra ayuda sobre los parámetros y opciones disponibles
    """
    print("=" * 70)
    print("📚 GUÍA DE PARÁMETROS - GENERADOR DE ARTE EULER 2D")
    print("=" * 70)
    
    print("\n🖼️  DIMENSIONES Y RESOLUCIÓN:")
    print("   • Ancho/Alto: Tamaño final de la imagen en píxeles")
    print("   • Resolución: Densidad de cálculo (mayor = más detalle, más lento)")
    print("   • Recomendado: 600x600 con resolución 200 para balance calidad/velocidad")
    
    print("\n🌊 TIPOS DE FLUJO:")
    print("   • vortex_dance: Vórtices que interactúan, patrones circulares")
    print("   • spiral_galaxy: Patrones espirales, brazos galácticos")
    print("   • turbulent_ocean: Flujo caótico, texturas orgánicas")
    
    print("\n🎨 ESTILOS DE ARTE:")
    print("   • mixed_media: Combina vorticidad y velocidad (más artístico)")
    print("   • vorticity_flow: Basado en rotación del fluido (patrones definidos)")
    print("   • velocity_field: Basado en velocidad del fluido (flujos direccionales)")
    
    print("\n⚙️  PARÁMETROS FÍSICOS:")
    print("   • Viscosidad (0.001-0.01): Menor = más definido, Mayor = más suave")
    print("   • Paso temporal (0.001-0.01): Menor = más preciso, Mayor = más rápido")
    print("   • Pasos de simulación: Más pasos = mayor evolución del patrón")
    
    print("\n🌈 MAPAS DE COLORES:")
    print("   • HSV: Multicolor completo, muy vibrante")
    print("   • Plasma: Morado-rosa-amarillo, elegante")
    print("   • Viridis: Verde-azul, científico")
    print("   • Inferno: Negro-rojo-amarillo, dramático")
    print("   • Magma: Negro-morado-blanco, sofisticado")
    
    print("\n💡 CONSEJOS:")
    print("   • Para arte detallado: Aumenta resolución y pasos")
    print("   • Para arte suave: Aumenta viscosidad")
    print("   • Para colores vibrantes: Usa HSV o Plasma")
    print("   • Para animaciones: Usa menos resolución para mejor rendimiento")
    
    print("=" * 70)


# Punto de entrada principal
if __name__ == "__main__":
    print("=" * 70)
    print("🎨 GENERADOR DE ARTE PROCEDURAL INTERACTIVO")
    print("   Ecuaciones de Euler 2D - Versión Personalizable")
    print("=" * 70)
    
    print("\n¿Qué deseas hacer?")
    print("1. Crear arte personalizado (interactivo)")
    print("2. Ver ejemplos predefinidos")
    print("3. Mostrar ayuda y guía de parámetros")
    print("4. Salir")
    
    while True:
        choice = input("\nSelecciona una opción [1]: ") or "1"
        
        if choice == "1":
            main()
            break
        elif choice == "2":
            create_preset_examples()
            break
        elif choice == "3":
            show_help()
            print("\n¿Deseas crear arte ahora? (s/n) [s]: ", end="")
            if input().lower() != 'n':
                main()
            break
        elif choice == "4":
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida, intenta de nuevo")
    
    print("\n🎨 ¡Gracias por usar el Generador de Arte Euler 2D!")
