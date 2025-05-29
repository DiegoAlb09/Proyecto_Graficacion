import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.integrate import ode
import colorsys

class EulerArtGenerator:
    """
    Generador de Arte Procedural basado en las Ecuaciones de Euler en 2D
    VERSIÓN MEJORADA PARA IMÁGENES MÁS NÍTIDAS
    """
    
    def __init__(self, width=800, height=600, resolution=200):  # AUMENTADA resolución por defecto
        """
        Inicializa el generador de arte
        
        Args:
            width (int): Ancho del canvas en píxeles
            height (int): Alto del canvas en píxeles  
            resolution (int): Resolución de la grilla computacional (AUMENTADA para menos blur)
        """
        self.width = width
        self.height = height
        
        # CAMBIO 1: Permitir resoluciones más altas sin limitar tanto
        max_safe_resolution = min(width, height)  # Removido el //2 que limitaba demasiado
        if resolution > max_safe_resolution:
            print(f"Warning: Resolution {resolution} is too large for dimensions {width}x{height}.")
            print(f"Adjusting resolution to {max_safe_resolution}.")
            resolution = max_safe_resolution
        
        self.resolution = resolution
        
        # Crear grilla espacial para los cálculos
        self.x = np.linspace(0, width, resolution)
        self.y = np.linspace(0, height, resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # CAMBIO 2: Parámetros físicos ajustados para mejor definición
        self.viscosity = 0.005   # REDUCIDA para menos difusión = menos blur
        self.dt = 0.005          # REDUCIDO para mayor precisión temporal
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
        
    def initialize_flow_field(self, flow_type='vortex_dance'):
        """
        Inicializa el campo de flujo con diferentes patrones
        MEJORADO: Patrones más definidos y menos difusos
        """
        if flow_type == 'vortex_dance':
            # CAMBIO 3: Vórtices más concentrados y definidos
            centers = [(0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)]
            strengths = [5.0, -4.0, 3.0, -4.5]  # Fuerzas más altas para mejor definición
            
            self.force_centers = [(cx * self.width, cy * self.height) for cx, cy in centers]
            self.force_strengths = strengths.copy()
            
            for (cx, cy), strength in zip(centers, strengths):
                cx_abs, cy_abs = cx * self.width, cy * self.height
                
                dx = self.X - cx_abs
                dy = self.Y - cy_abs
                r_squared = dx**2 + dy**2 + 10  # CAMBIO: Núcleo más pequeño = vórtices más definidos
                
                # Función de decaimiento más pronunciada
                decay = np.exp(-r_squared / (0.15 * min(self.width, self.height)**2))
                
                self.u += -strength * dy / r_squared * decay
                self.v += strength * dx / r_squared * decay
                
        elif flow_type == 'spiral_galaxy':
            # CAMBIO 4: Espiral más definida
            center_x, center_y = self.width/2, self.height/2
            dx = self.X - center_x
            dy = self.Y - center_y
            r = np.sqrt(dx**2 + dy**2) + 1e-6
            theta = np.arctan2(dy, dx)
            
            # Parámetros ajustados para mayor definición
            v_r = 0.8 * r * np.sin(8 * theta) * np.exp(-r / (0.3 * min(self.width, self.height)))
            v_theta = 6.0 / (1 + r/50) * np.exp(-r / (0.4 * min(self.width, self.height)))
            
            self.u = v_r * np.cos(theta) - v_theta * np.sin(theta)
            self.v = v_r * np.sin(theta) + v_theta * np.cos(theta)
            
        elif flow_type == 'turbulent_ocean':
            # CAMBIO 5: Turbulencia más estructurada
            np.random.seed(42)
            n_modes = 8  # Menos modos pero más definidos
            
            for i in range(n_modes):
                kx = np.random.uniform(-0.08, 0.08)  # Frecuencias más bajas = menos blur
                ky = np.random.uniform(-0.08, 0.08)
                amplitude = np.random.uniform(2.0, 4.0)  # Amplitudes más altas
                phase = np.random.uniform(0, 2*np.pi)
                
                wave = amplitude * np.sin(kx * self.X + ky * self.Y + phase)
                self.u += np.gradient(wave, axis=1) * 0.5
                self.v += np.gradient(wave, axis=0) * 0.5
                
        # Calcular vorticidad inicial
        self.calculate_vorticity()
    
    def calculate_vorticity(self):
        """
        Calcula la vorticidad del campo de velocidad
        MEJORADO: Cálculo más preciso
        """
        # CAMBIO 6: Usar espaciado de grilla apropiado para gradientes más precisos
        dx = self.width / self.resolution
        dy = self.height / self.resolution
        
        du_dy = np.gradient(self.u, dy, axis=0)
        dv_dx = np.gradient(self.v, dx, axis=1)
        self.vorticity = np.clip(dv_dx - du_dy, -500, 500)  # Rango más conservador
    
    def advect_field(self, field, u, v, dt):
        """
        Advecta un campo escalar usando método más preciso
        MEJORADO: Mayor precisión = menos blur
        """
        # CAMBIO 7: Clips más conservadores para mantener estructura
        u_safe = np.clip(u, -50, 50)
        v_safe = np.clip(v, -50, 50)
        field_safe = np.nan_to_num(field, nan=0.0, posinf=50.0, neginf=-50.0)
        
        # Espaciado de grilla para gradientes más precisos
        dx = self.width / self.resolution
        dy = self.height / self.resolution
        
        # Gradientes con espaciado correcto
        dudx = np.gradient(field_safe, dx, axis=1)
        dudy = np.gradient(field_safe, dy, axis=0)
        
        dudx = np.clip(dudx, -50, 50)
        dudy = np.clip(dudy, -50, 50)
        
        # Método Runge-Kutta de orden 2 para mayor precisión
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
        """
        Aplica difusión viscosa con menor blur
        CAMBIO 8: Viscosidad reducida y más controlada
        """
        field_safe = np.nan_to_num(field, nan=0.0, posinf=50.0, neginf=-50.0)
        
        # Laplaciano más preciso con espaciado correcto
        dx = self.width / self.resolution
        dy = self.height / self.resolution
        
        # Laplaciano de 2º orden más preciso
        d2_dx2 = (np.roll(field_safe, 1, axis=1) - 2*field_safe + np.roll(field_safe, -1, axis=1)) / (dx**2)
        d2_dy2 = (np.roll(field_safe, 1, axis=0) - 2*field_safe + np.roll(field_safe, -1, axis=0)) / (dy**2)
        
        laplacian = d2_dx2 + d2_dy2
        laplacian = np.clip(laplacian, -100, 100)
        
        result = field_safe + viscosity * dt * laplacian
        
        return np.clip(np.nan_to_num(result, nan=0.0), -500, 500)
    
    def add_dynamic_forces(self):
        """
        Añade fuerzas externas más definidas
        """
        t = self.time * 0.03  # Evolución más lenta para mejor visualización
        
        # CAMBIO 9: Fuerzas más localizadas = menos blur
        for i in range(2):  # Menos fuerzas para mayor claridad
            angle = t + i * np.pi
            radius = 0.15 * min(self.width, self.height)
            
            center_x = self.width/2 + radius * np.cos(angle)
            center_y = self.height/2 + radius * np.sin(angle)
            strength = 3.0 * np.cos(t + i)
            
            dx = self.X - center_x
            dy = self.Y - center_y
            r_squared = dx**2 + dy**2 + 25  # Núcleo más grande para suavidad controlada
            
            # Función de decaimiento más pronunciada
            decay = np.exp(-r_squared / (0.1 * min(self.width, self.height)**2))
            
            self.u += -strength * dy / r_squared * decay * 0.05
            self.v += strength * dx / r_squared * decay * 0.05
    
    def step_simulation(self):
        """
        Avanza la simulación con mayor precisión
        """
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
        
        # CAMBIO 10: Menos ruido aleatorio para mayor claridad
        if self.time % 10 == 0:
            noise_strength = 0.3  # Reducido
            self.u += noise_strength * np.random.randn(*self.u.shape) * 0.02
            self.v += noise_strength * np.random.randn(*self.v.shape) * 0.02
        
        # Inyección de energía más controlada
        if self.time % 20 == 0:
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
    
    def field_to_color(self, field, colormap='hsv', normalize=True, enhance_contrast=True):
        """
        Convierte un campo escalar a colores RGB con mayor contraste
        CAMBIO 11: Mejor mapeo de colores para mayor definición
        """
        field_safe = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
        
        if normalize:
            field_min = np.percentile(field_safe, 5)   # Usar percentiles para mejor contraste
            field_max = np.percentile(field_safe, 95)
            
            if np.abs(field_max - field_min) < 1e-6:
                field_min = 0.0
                field_max = 1.0
            
            field_norm = np.clip((field_safe - field_min) / (field_max - field_min + 1e-8), 0, 1)
            
            # CAMBIO 12: Mejorar contraste con función gamma
            if enhance_contrast:
                field_norm = np.power(field_norm, 0.8)  # Gamma correction para más contraste
        else:
            field_norm = np.clip(field_safe, 0, 1)
        
        if colormap == 'hsv':
            h = field_norm
            s = np.ones_like(h) * 0.98  # Saturación máxima
            v = 0.3 + 0.7 * field_norm  # Valor variable para más contraste
            
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
    
    def velocity_to_color(self, enhance_contrast=True):
        """
        Convierte el campo de velocidad a colores con mayor definición
        """
        u_safe = np.nan_to_num(self.u, nan=0.0)
        v_safe = np.nan_to_num(self.v, nan=0.0)
        
        velocity_mag = np.sqrt(u_safe**2 + v_safe**2)
        velocity_angle = np.arctan2(v_safe, u_safe)
        
        hue = (velocity_angle + np.pi) / (2 * np.pi)
        
        # CAMBIO 13: Mejor mapeo de saturación para mayor contraste
        max_vel = np.percentile(velocity_mag, 90)  # Percentil 90 para mejor contraste
        if max_vel < 1e-6:
            max_vel = 1.0
            
        saturation = np.clip(velocity_mag / (max_vel + 1e-8), 0, 1)
        
        # Mejorar contraste de saturación
        if enhance_contrast:
            saturation = np.power(saturation, 0.7)  # Gamma correction
        
        value = 0.4 + 0.6 * saturation  # Valor variable para más definición
        
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
    
    def create_art_frame(self, art_style='vorticity_flow'):
        """
        Genera un frame de arte con mayor definición
        """
        if art_style == 'vorticity_flow':
            colors = self.field_to_color(self.vorticity, 'hsv', enhance_contrast=True)
            
        elif art_style == 'velocity_field':
            colors = self.velocity_to_color(enhance_contrast=True)
            
        elif art_style == 'mixed_media':
            vort_colors = self.field_to_color(self.vorticity, 'plasma', enhance_contrast=True)
            vel_colors = self.velocity_to_color(enhance_contrast=True)
            
            vort_colors = np.nan_to_num(vort_colors, nan=0.0)
            vel_colors = np.nan_to_num(vel_colors, nan=0.0)
            
            alpha = 0.6 + 0.2 * np.sin(self.time * 0.05)  # Mezcla más lenta
            colors = alpha * vort_colors + (1 - alpha) * vel_colors
            colors = np.nan_to_num(colors, nan=0.0)
        
        # CAMBIO 14: Usar interpolación de mayor calidad para el redimensionado
        from scipy.ndimage import zoom
        
        zoom_factor_y = self.height / colors.shape[0]
        zoom_factor_x = self.width / colors.shape[1]
        
        # Usar interpolación de orden superior para menos blur
        if zoom_factor_x > 1 or zoom_factor_y > 1:
            # Si estamos escalando hacia arriba, usar interpolación cúbica
            colors_resized = zoom(colors, (zoom_factor_y, zoom_factor_x, 1), order=3)
        else:
            # Si estamos escalando hacia abajo, usar interpolación lineal
            colors_resized = zoom(colors, (zoom_factor_y, zoom_factor_x, 1), order=1)
        
        return np.clip(colors_resized, 0, 1)
    
    def generate_static_art(self, steps=300, flow_type='vortex_dance', 
                           art_style='mixed_media', save_path=None, dpi=600):
        """
        Genera una obra de arte estática con mayor resolución
        CAMBIO 15: DPI más alto por defecto para mayor calidad
        """
        print(f"Inicializando flujo tipo: {flow_type}")
        self.initialize_flow_field(flow_type)
        
        print(f"Simulando {steps} pasos temporales...")
        for i in range(steps):
            if i % 50 == 0:
                print(f"Paso {i}/{steps}")
            self.step_simulation()
        
        print(f"Generando arte con estilo: {art_style}")
        final_art = self.create_art_frame(art_style)
        
        if save_path:
            plt.figure(figsize=(12, 9))
            plt.imshow(final_art, interpolation='bilinear')  # Interpolación suave pero no borrosa
            plt.axis('off')
            plt.title(f'Arte Procedural HD - Ecuaciones de Euler 2D\n'
                     f'Flujo: {flow_type}, Estilo: {art_style}', 
                     fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')  # Fondo negro para mayor contraste
            plt.show()
            print(f"Arte HD guardado en: {save_path}")
        
        return final_art
    
    def create_animation(self, frames=100, flow_type='vortex_dance', 
                        art_style='mixed_media', save_path=None):
        """
        Crea una animación de alta calidad
        """
        print(f"Creando animación HD con {frames} frames...")
        
        self.initialize_flow_field(flow_type)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.axis('off')
        fig.patch.set_facecolor('black')  # Fondo negro
        fig.tight_layout()  # Asegurar espaciado correcto antes de la animación
        
        art_frame = self.create_art_frame(art_style)
        im = ax.imshow(art_frame, extent=[0, self.width, 0, self.height], 
                      interpolation='bilinear')  # Interpolación de calidad
        
        def animate(frame):
            for _ in range(3):  # Menos pasos por frame para mayor control
                self.step_simulation()
            
            art_frame = self.create_art_frame(art_style)
            im.set_array(art_frame)
            
            ax.set_title(f'Arte Procedural HD - Frame {frame}\n'
                        f'Ecuaciones de Euler 2D (Tiempo: {self.time})', 
                        fontsize=12, color='white')
            
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=frames, 
                           interval=100, blit=True, repeat=True)
        
        if save_path:
            print(f"Guardando animación HD en: {save_path}")
            # CAMBIO 16: Configuración mejorada para GIF de alta calidad
            anim.save(save_path, writer='pillow', fps=10, 
                     savefig_kwargs={'facecolor': 'black'})
        
        plt.show()
        return anim

# Ejemplo de uso con configuración HD
if __name__ == "__main__":
    # CAMBIO 17: Configuración por defecto para mayor calidad
    generator = EulerArtGenerator(width=800, height=800, resolution=300)  # Resolución muy alta
    
    print("=== GENERADOR DE ARTE PROCEDURAL HD - SIN BLUR ===\n")
    
    # Arte estático HD
    print("1. Generando arte HD: Danza de Vórtices")
    art1 = generator.generate_static_art(
        steps=200,
        flow_type='vortex_dance',
        art_style='mixed_media',
        save_path='euler_art_HD_vortex.png',
        dpi=600  # DPI muy alto
    )
    
    # Nuevo generador para arte diferente
    generator = EulerArtGenerator(width=800, height=800, resolution=300)
    
    print("2. Generando arte HD: Galaxia Espiral")
    art2 = generator.generate_static_art(
        steps=250,
        flow_type='spiral_galaxy',
        art_style='velocity_field',
        save_path='euler_art_HD_spiral.png',
        dpi=600
    )
    
    # Nuevo generador para animación HD
    generator = EulerArtGenerator(width=600, height=600, resolution=200)  # Resolución balanceada para animación
    
    print("3. Creando animación HD...")
    animation = generator.create_animation(
        frames=80,
        flow_type='vortex_dance',
        art_style='mixed_media',
        save_path='euler_art_HD_animation.gif'
    )
    
    print("\n=== ARTE HD GENERADO SIN BLUR ===")
