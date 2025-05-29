import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from scipy.integrate import ode
import colorsys

class EulerArtGenerator:
    """
    Generador de Arte Procedural basado en las Ecuaciones de Euler en 2D
    
    Este generador crea arte visual mediante la simulación de campos de flujo
    usando las ecuaciones de Euler para fluidos incompresibles en 2D.
    """
    
    def __init__(self, width=800, height=600, resolution=100):
        """
        Inicializa el generador de arte
        
        Args:
            width (int): Ancho del canvas en píxeles
            height (int): Alto del canvas en píxeles  
            resolution (int): Resolución de la grilla computacional
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # Crear grilla espacial para los cálculos
        self.x = np.linspace(0, width, resolution)
        self.y = np.linspace(0, height, resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Parámetros físicos del flujo
        self.viscosity = 0.01  # Viscosidad del fluido
        self.dt = 0.01         # Paso temporal para integración
        self.time = 0          # Tiempo actual de simulación
        
        # Campos de velocidad (componentes u, v)
        self.u = np.zeros_like(self.X)  # Velocidad en x
        self.v = np.zeros_like(self.Y)  # Velocidad en y
        
        # Campo de vorticidad (rotacional de la velocidad)
        self.vorticity = np.zeros_like(self.X)
        
        # Canvas para el arte final
        self.canvas = np.zeros((height, width, 3))
        
    def initialize_flow_field(self, flow_type='vortex_dance'):
        """
        Inicializa el campo de flujo con diferentes patrones
        
        Args:
            flow_type (str): Tipo de flujo inicial
                - 'vortex_dance': Múltiples vórtices danzantes
                - 'spiral_galaxy': Espiral tipo galaxia
                - 'turbulent_ocean': Turbulencia oceánica
                - 'wind_patterns': Patrones de viento
        """
        if flow_type == 'vortex_dance':
            # Múltiples vórtices con diferentes fuerzas y posiciones
            centers = [(0.3, 0.3), (0.7, 0.7), (0.2, 0.8), (0.8, 0.2)]
            strengths = [2.0, -1.5, 1.0, -2.5]
            
            for (cx, cy), strength in zip(centers, strengths):
                # Convertir coordenadas relativas a absolutas
                cx_abs, cy_abs = cx * self.width, cy * self.height
                
                # Calcular distancias desde cada punto del grid al centro del vórtice
                dx = self.X - cx_abs
                dy = self.Y - cy_abs
                r_squared = dx**2 + dy**2 + 1e-6  # Evitar división por cero
                
                # Añadir contribución del vórtice al campo de velocidad
                self.u += -strength * dy / r_squared
                self.v += strength * dx / r_squared
                
        elif flow_type == 'spiral_galaxy':
            # Campo tipo galaxia espiral
            center_x, center_y = self.width/2, self.height/2
            dx = self.X - center_x
            dy = self.Y - center_y
            r = np.sqrt(dx**2 + dy**2) + 1e-6
            theta = np.arctan2(dy, dx)
            
            # Velocidad radial y tangencial
            v_r = 0.1 * r * np.sin(3 * theta)
            v_theta = 2.0 / (1 + r/100)
            
            self.u = v_r * np.cos(theta) - v_theta * np.sin(theta)
            self.v = v_r * np.sin(theta) + v_theta * np.cos(theta)
            
        elif flow_type == 'turbulent_ocean':
            # Turbulencia pseudo-aleatoria
            np.random.seed(42)  # Para reproducibilidad
            n_modes = 8
            
            for i in range(n_modes):
                # Generar ondas con diferentes frecuencias y amplitudes
                kx = np.random.uniform(-0.1, 0.1)
                ky = np.random.uniform(-0.1, 0.1)
                amplitude = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2*np.pi)
                
                wave = amplitude * np.sin(kx * self.X + ky * self.Y + phase)
                self.u += np.gradient(wave, axis=1)
                self.v += np.gradient(wave, axis=0)
                
        # Calcular vorticidad inicial
        self.calculate_vorticity()
    
    def calculate_vorticity(self):
        """
        Calcula la vorticidad del campo de velocidad
        Vorticidad = ∂v/∂x - ∂u/∂y
        """
        # Usar diferencias finitas para calcular gradientes
        du_dy = np.gradient(self.u, axis=0)
        dv_dx = np.gradient(self.v, axis=1)
        self.vorticity = dv_dx - du_dy
    
    def advect_field(self, field, u, v, dt):
        """
        Advecta un campo escalar usando el método de MacCormack
        
        Args:
            field: Campo escalar a advectar
            u, v: Componentes de velocidad
            dt: Paso temporal
            
        Returns:
            Campo advectado
        """
        # Paso predictor (Euler hacia adelante)
        dudx = np.gradient(field, axis=1)
        dudy = np.gradient(field, axis=0)
        
        field_pred = field - dt * (u * dudx + v * dudy)
        
        # Paso corrector (promedio con Euler hacia atrás)
        dudx_pred = np.gradient(field_pred, axis=1)
        dudy_pred = np.gradient(field_pred, axis=0)
        
        field_corr = field_pred - dt * (u * dudx_pred + v * dudy_pred)
        
        # Promediar predictor y corrector
        return 0.5 * (field_pred + field_corr)
    
    def apply_viscosity(self, field, viscosity, dt):
        """
        Aplica difusión viscosa usando el operador Laplaciano
        
        Args:
            field: Campo a difundir
            viscosity: Coeficiente de viscosidad
            dt: Paso temporal
            
        Returns:
            Campo después de la difusión
        """
        # Calcular Laplaciano usando diferencias finitas
        laplacian = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                    np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 
                    4 * field)
        
        return field + viscosity * dt * laplacian
    
    def step_simulation(self):
        """
        Avanza la simulación un paso temporal usando las ecuaciones de Euler
        """
        # 1. Advección: transportar la vorticidad con la velocidad
        self.vorticity = self.advect_field(self.vorticity, self.u, self.v, self.dt)
        
        # 2. Difusión viscosa
        self.vorticity = self.apply_viscosity(self.vorticity, self.viscosity, self.dt)
        
        # 3. Reconstruir velocidad desde vorticidad (método simplificado)
        # En una implementación completa se usaría el método de proyección
        self.u = self.apply_viscosity(self.u, self.viscosity, self.dt)
        self.v = self.apply_viscosity(self.v, self.viscosity, self.dt)
        
        # 4. Añadir pequeñas perturbaciones para mantener la dinámica
        if self.time % 20 == 0:  # Más frecuente para mayor dinamismo
            noise_strength = 0.3  # Mayor intensidad
            self.u += noise_strength * np.random.randn(*self.u.shape) * 0.02
            self.v += noise_strength * np.random.randn(*self.v.shape) * 0.02
        
        # 5. Aplicar condiciones de frontera (velocidad cero en los bordes)
        self.u[0, :] = self.u[-1, :] = 0
        self.u[:, 0] = self.u[:, -1] = 0
        self.v[0, :] = self.v[-1, :] = 0
        self.v[:, 0] = self.v[:, -1] = 0
        
        self.time += 1
    
    def field_to_color(self, field, colormap='hsv', normalize=True):
        """
        Convierte un campo escalar a colores RGB
        
        Args:
            field: Campo escalar
            colormap: Esquema de colores ('hsv', 'plasma', 'viridis')
            normalize: Si normalizar el campo al rango [0,1]
            
        Returns:
            Array de colores RGB
        """
        if normalize:
            # Normalizar el campo al rango [0, 1]
            field_norm = (field - field.min()) / (field.max() - field.min() + 1e-8)
        else:
            field_norm = np.clip(field, 0, 1)
        
        if colormap == 'hsv':
            # Usar el campo como matiz (hue) en HSV
            h = field_norm
            s = np.ones_like(h) * 0.8  # Saturación alta
            v = np.ones_like(h) * 0.9  # Valor alto
            
            # Convertir HSV a RGB
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
    
    def velocity_to_color(self):
        """
        Convierte el campo de velocidad a colores usando magnitud y dirección
        
        Returns:
            Array de colores RGB
        """
        # Magnitud de la velocidad
        velocity_mag = np.sqrt(self.u**2 + self.v**2)
        
        # Dirección de la velocidad (ángulo)
        velocity_angle = np.arctan2(self.v, self.u)
        
        # Normalizar ángulo a [0, 1] para usar como matiz
        hue = (velocity_angle + np.pi) / (2 * np.pi)
        
        # Usar magnitud para saturación y valor
        saturation = np.clip(velocity_mag / np.max(velocity_mag + 1e-8), 0, 1)
        value = np.ones_like(hue) * 0.9
        
        # Convertir HSV a RGB píxel por píxel
        rgb = np.zeros((*self.X.shape, 3))
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                rgb[i, j] = colorsys.hsv_to_rgb(hue[i, j], saturation[i, j], value[i, j])
        
        return rgb
    
    def create_art_frame(self, art_style='vorticity_flow'):
        """
        Genera un frame de arte basado en el estado actual de la simulación
        
        Args:
            art_style: Estilo de arte a generar
                - 'vorticity_flow': Basado en vorticidad
                - 'velocity_field': Basado en velocidad
                - 'streamlines': Líneas de corriente artísticas
                - 'mixed_media': Combinación de múltiples campos
        """
        if art_style == 'vorticity_flow':
            # Arte basado en la vorticidad
            colors = self.field_to_color(self.vorticity, 'hsv')
            
        elif art_style == 'velocity_field':
            # Arte basado en el campo de velocidad
            colors = self.velocity_to_color()
            
        elif art_style == 'mixed_media':
            # Combinación artística de diferentes campos
            vort_colors = self.field_to_color(self.vorticity, 'plasma')
            vel_colors = self.velocity_to_color()
            
            # Mezclar colores con pesos dinámicos
            alpha = 0.6
            colors = alpha * vort_colors + (1 - alpha) * vel_colors
            
        # Redimensionar a la resolución del canvas final
        from scipy.ndimage import zoom
        zoom_factor_y = self.height / colors.shape[0]
        zoom_factor_x = self.width / colors.shape[1]
        
        colors_resized = zoom(colors, (zoom_factor_y, zoom_factor_x, 1), order=1)
        
        return np.clip(colors_resized, 0, 1)
    
    def generate_static_art(self, steps=200, flow_type='vortex_dance', 
                           art_style='mixed_media', save_path=None):
        """
        Genera una obra de arte estática
        
        Args:
            steps: Número de pasos de simulación
            flow_type: Tipo de flujo inicial
            art_style: Estilo de renderizado
            save_path: Ruta para guardar la imagen (opcional)
            
        Returns:
            Array de imagen RGB final
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
            plt.imshow(final_art)
            plt.axis('off')
            plt.title(f'Arte Procedural - Ecuaciones de Euler 2D\n'
                     f'Flujo: {flow_type}, Estilo: {art_style}', 
                     fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Arte guardado en: {save_path}")
        
        return final_art
    
    def create_animation(self, frames=100, flow_type='vortex_dance', 
                        art_style='mixed_media', save_path=None):
        """
        Crea una animación del arte procedural
        
        Args:
            frames: Número de frames de la animación
            flow_type: Tipo de flujo inicial
            art_style: Estilo de renderizado
            save_path: Ruta para guardar el GIF (opcional)
        """
        print(f"Creando animación con {frames} frames...")
        
        self.initialize_flow_field(flow_type)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.axis('off')
        fig.tight_layout()  # Asegurar espaciado correcto antes de la animación
        
        # Imagen inicial
        art_frame = self.create_art_frame(art_style)
        im = ax.imshow(art_frame, extent=[0, self.width, 0, self.height])
        
        def animate(frame):
            """Función de animación para cada frame"""
            # Avanzar simulación
            for _ in range(2):  # Reducir pasos para ver evolución más gradual
                self.step_simulation()
            
            # Actualizar arte
            art_frame = self.create_art_frame(art_style)
            im.set_array(art_frame)
            
            ax.set_title(f'Arte Procedural Dinámico - Frame {frame}\n'
                        f'Ecuaciones de Euler 2D', fontsize=12)
            
            return [im]
        
        # Crear animación
        anim = FuncAnimation(fig, animate, frames=frames, 
                           interval=100, blit=True, repeat=True)
        
        if save_path:
            print(f"Guardando animación en: {save_path}")
            anim.save(save_path, writer='pillow', fps=10)
        
        plt.show()
        return anim

# Ejemplo de uso del generador
if __name__ == "__main__":
    # Crear instancia del generador
    generator = EulerArtGenerator(width=600, height=600, resolution=80)
    
    # Generar diferentes tipos de arte
    print("=== GENERADOR DE ARTE PROCEDURAL CON ECUACIONES DE EULER ===\n")
    
    # Arte estático - Danza de Vórtices
    print("1. Generando arte: Danza de Vórtices")
    art1 = generator.generate_static_art(
        steps=150,
        flow_type='vortex_dance',
        art_style='mixed_media',
        save_path='euler_art_vortex_dance.png'
    )
    
    # Resetear para nuevo arte
    generator = EulerArtGenerator(width=600, height=600, resolution=80)
    
    # Arte estático - Galaxia Espiral
    print("2. Generando arte: Galaxia Espiral")
    art2 = generator.generate_static_art(
        steps=200,
        flow_type='spiral_galaxy',
        art_style='velocity_field',
        save_path='euler_art_spiral_galaxy.png'
    )
    
    # Crear animación (descomenta para ejecutar)
    print("3. Creando animación...")
    generator_anim = EulerArtGenerator(width=400, height=400, resolution=60)
    animation = generator_anim.create_animation(
        frames=50,
        flow_type='turbulent_ocean',
        art_style='mixed_media',
        save_path='euler_art_animation.gif'
    )
    
    print("\n=== ARTE GENERADO EXITOSAMENTE ===")
