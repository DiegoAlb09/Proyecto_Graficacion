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
        
        # Ensure resolution is appropriate for the dimensions
        # Resolution should be less than the minimum of width and height
        max_safe_resolution = min(width, height) // 2
        if resolution > max_safe_resolution:
            print(f"Warning: Resolution {resolution} is too large for dimensions {width}x{height}.")
            print(f"Adjusting resolution to {max_safe_resolution} to avoid division by zero errors.")
            resolution = max_safe_resolution
        
        self.resolution = resolution
        
        # Crear grilla espacial para los cálculos
        self.x = np.linspace(0, width, resolution)
        self.y = np.linspace(0, height, resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Parámetros físicos del flujo - MODIFICADOS para mayor dinamismo y estabilidad
        self.viscosity = 0.01   # Aumentada para mayor estabilidad
        self.dt = 0.01          # Reducido para mayor estabilidad
        self.time = 0          # Tiempo actual de simulación
        
        # Campos de velocidad (componentes u, v)
        self.u = np.zeros_like(self.X)  # Velocidad en x
        self.v = np.zeros_like(self.Y)  # Velocidad en y
        
        # Campo de vorticidad (rotacional de la velocidad)
        self.vorticity = np.zeros_like(self.X)
        
        # Canvas para el arte final
        self.canvas = np.zeros((height, width, 3))
        
        # NUEVO: Variables para fuerzas externas dinámicas
        self.force_centers = []
        self.force_strengths = []
        self.force_evolution_rate = 0.1
        
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
            strengths = [3.0, -2.5, 1.5, -3.5]  # Incrementadas las fuerzas
            
            # NUEVO: Guardar centros para animación dinámica
            self.force_centers = [(cx * self.width, cy * self.height) for cx, cy in centers]
            self.force_strengths = strengths.copy()
            
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
            # Campo tipo galaxia espiral - MODIFICADO para mayor dinamismo
            center_x, center_y = self.width/2, self.height/2
            dx = self.X - center_x
            dy = self.Y - center_y
            r = np.sqrt(dx**2 + dy**2) + 1e-6
            theta = np.arctan2(dy, dx)
            
            # Velocidad radial y tangencial con mayor amplitud
            v_r = 0.3 * r * np.sin(5 * theta)  # Incrementado
            v_theta = 4.0 / (1 + r/80)         # Incrementado
            
            self.u = v_r * np.cos(theta) - v_theta * np.sin(theta)
            self.v = v_r * np.sin(theta) + v_theta * np.cos(theta)
            
            # Guardar parámetros para evolución dinámica
            self.spiral_time_factor = 0
            
        elif flow_type == 'turbulent_ocean':
            # Turbulencia pseudo-aleatoria con mayor intensidad
            np.random.seed(42)  # Para reproducibilidad
            n_modes = 12  # Más modos para mayor complejidad
            
            for i in range(n_modes):
                # Generar ondas con diferentes frecuencias y amplitudes
                kx = np.random.uniform(-0.15, 0.15)  # Mayor rango
                ky = np.random.uniform(-0.15, 0.15)
                amplitude = np.random.uniform(1.0, 3.0)  # Mayor amplitud
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
        self.vorticity = np.clip(dv_dx - du_dy, -1000, 1000)  # Clip to reasonable range
    
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
        # Clip velocity fields to prevent numerical instability
        u_safe = np.clip(u, -100, 100)
        v_safe = np.clip(v, -100, 100)
        
        # Ensure field doesn't have NaN values
        field_safe = np.nan_to_num(field, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Paso predictor (Euler hacia adelante)
        dudx = np.gradient(field_safe, axis=1)
        dudy = np.gradient(field_safe, axis=0)
        
        # Clip gradients to prevent explosion
        dudx = np.clip(dudx, -100, 100)
        dudy = np.clip(dudy, -100, 100)
        
        field_pred = field_safe - dt * (u_safe * dudx + v_safe * dudy)
        field_pred = np.nan_to_num(field_pred, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Paso corrector (promedio con Euler hacia atrás)
        dudx_pred = np.gradient(field_pred, axis=1)
        dudy_pred = np.gradient(field_pred, axis=0)
        
        # Clip gradients again
        dudx_pred = np.clip(dudx_pred, -100, 100)
        dudy_pred = np.clip(dudy_pred, -100, 100)
        
        field_corr = field_pred - dt * (u_safe * dudx_pred + v_safe * dudy_pred)
        field_corr = np.nan_to_num(field_corr, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Promediar predictor y corrector
        result = 0.5 * (field_pred + field_corr)
        
        # Final safety check
        return np.clip(np.nan_to_num(result, nan=0.0), -1000, 1000)
    
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
        # Ensure field doesn't have NaN values
        field_safe = np.nan_to_num(field, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Calcular Laplaciano usando diferencias finitas
        laplacian = (np.roll(field_safe, 1, axis=0) + np.roll(field_safe, -1, axis=0) +
                    np.roll(field_safe, 1, axis=1) + np.roll(field_safe, -1, axis=1) - 
                    4 * field_safe)
        
        # Clip laplacian to avoid extreme values
        laplacian = np.clip(laplacian, -1000, 1000)
        
        result = field_safe + viscosity * dt * laplacian
        
        # Safety checks
        return np.clip(np.nan_to_num(result, nan=0.0), -1000, 1000)
    
    def add_dynamic_forces(self):
        """
        NUEVO: Añade fuerzas externas que evolucionan en el tiempo
        """
        # Fuerzas rotatorias que cambian de posición
        t = self.time * 0.05
        
        # Añadir vórtices que se mueven en círculos
        for i in range(3):
            angle = t + i * 2 * np.pi / 3
            radius = 0.2 * min(self.width, self.height)
            
            center_x = self.width/2 + radius * np.cos(angle)
            center_y = self.height/2 + radius * np.sin(angle)
            strength = 2.0 * np.sin(t + i)  # Fuerza que oscila
            
            # Calcular distancias
            dx = self.X - center_x
            dy = self.Y - center_y
            r_squared = dx**2 + dy**2 + 1e-6
            
            # Añadir fuerza del vórtice móvil
            self.u += -strength * dy / r_squared * 0.1
            self.v += strength * dx / r_squared * 0.1
    
    def step_simulation(self):
        """
        Avanza la simulación un paso temporal usando las ecuaciones de Euler
        """
        # Ensure velocity fields don't have NaN values
        self.u = np.nan_to_num(self.u, nan=0.0)
        self.v = np.nan_to_num(self.v, nan=0.0)
        
        # Clip velocity to reasonable bounds
        self.u = np.clip(self.u, -100, 100)
        self.v = np.clip(self.v, -100, 100)
        
        # 1. Añadir fuerzas dinámicas externas
        self.add_dynamic_forces()
        
        # 2. Advección: transportar la vorticidad con la velocidad
        self.vorticity = self.advect_field(self.vorticity, self.u, self.v, self.dt)
        
        # 3. Difusión viscosa (reducida para mantener más energía)
        self.vorticity = self.apply_viscosity(self.vorticity, self.viscosity, self.dt)
        
        # 4. Reconstruir velocidad desde vorticidad con mayor intensidad
        self.u = self.advect_field(self.u, self.u, self.v, self.dt)
        self.v = self.advect_field(self.v, self.u, self.v, self.dt)
        
        self.u = self.apply_viscosity(self.u, self.viscosity, self.dt)
        self.v = self.apply_viscosity(self.v, self.viscosity, self.dt)
        
        # 5. Añadir perturbaciones más frecuentes y fuertes
        if self.time % 5 == 0:  # Más frecuente
            noise_strength = 0.8  # Mayor intensidad
            self.u += noise_strength * np.random.randn(*self.u.shape) * 0.05
            self.v += noise_strength * np.random.randn(*self.v.shape) * 0.05
        
        # 6. Inyección periódica de energía en el centro
        if self.time % 10 == 0:
            center_x, center_y = self.width//2, self.height//2
            size = 10
            
            # Calculate grid spacing to avoid division by zero
            grid_spacing_x = max(1, self.width // self.resolution)
            grid_spacing_y = max(1, self.height // self.resolution)
            
            # Calculate grid coordinates safely
            cx = min(self.resolution - 3, max(2, center_x // grid_spacing_x))
            cy = min(self.resolution - 3, max(2, center_y // grid_spacing_y))
            
            # Now we can safely add energy to the center region
            self.u[cy-2:cy+3, cx-2:cx+3] += np.random.randn(5, 5) * 1.0
            self.v[cy-2:cy+3, cx-2:cx+3] += np.random.randn(5, 5) * 1.0
        
        # 7. Aplicar condiciones de frontera (velocidad cero en los bordes)
        self.u[0, :] = self.u[-1, :] = 0
        self.u[:, 0] = self.u[:, -1] = 0
        self.v[0, :] = self.v[-1, :] = 0
        self.v[:, 0] = self.v[:, -1] = 0
        
        # 8. Recalcular vorticidad
        self.calculate_vorticity()
        
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
        # First ensure field doesn't have NaN or infinite values
        field_safe = np.nan_to_num(field, nan=0.0, posinf=1.0, neginf=0.0)
        
        if normalize:
            # Normalize even if all values are the same to avoid division by zero
            field_min = np.min(field_safe)
            field_max = np.max(field_safe)
            
            # If min and max are too close, set default range
            if np.abs(field_max - field_min) < 1e-6:
                field_min = 0.0
                field_max = 1.0
            
            # Normalizar el campo al rango [0, 1] con mayor contraste
            field_norm = np.clip((field_safe - field_min) / (field_max - field_min + 1e-8), 0, 1)
        else:
            field_norm = np.clip(field_safe, 0, 1)
        
        if colormap == 'hsv':
            # Usar el campo como matiz (hue) en HSV con mayor saturación
            h = field_norm
            s = np.ones_like(h) * 0.95  # Saturación muy alta
            v = np.ones_like(h) * 0.95  # Valor alto
            
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
        # Ensure velocity fields don't have NaN values
        u_safe = np.nan_to_num(self.u, nan=0.0)
        v_safe = np.nan_to_num(self.v, nan=0.0)
        
        # Magnitud de la velocidad
        velocity_mag = np.sqrt(u_safe**2 + v_safe**2)
        
        # Dirección de la velocidad (ángulo)
        velocity_angle = np.arctan2(v_safe, u_safe)
        
        # Normalizar ángulo a [0, 1] para usar como matiz
        hue = (velocity_angle + np.pi) / (2 * np.pi)
        
        # Usar magnitud para saturación con mayor contraste
        max_vel = np.percentile(velocity_mag, 95)  # Usar percentil para mejor contraste
        if max_vel < 1e-6:  # Avoid division by zero
            max_vel = 1.0
            
        saturation = np.clip(velocity_mag / (max_vel + 1e-8), 0, 1)
        value = np.ones_like(hue) * 0.9
        
        # Convertir HSV a RGB píxel por píxel
        rgb = np.zeros((*self.X.shape, 3))
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                # Check for NaN before conversion
                h = hue[i, j]
                s = saturation[i, j]
                v = value[i, j]
                
                # Safety check - replace NaN or out-of-range values
                if np.isnan(h) or np.isnan(s) or np.isnan(v) or h < 0 or h > 1 or s < 0 or s > 1 or v < 0 or v > 1:
                    h = 0.0
                    s = 0.0
                    v = 0.9
                    
                rgb[i, j] = colorsys.hsv_to_rgb(h, s, v)
        
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
            
            # Check for NaN values in color arrays
            vort_colors = np.nan_to_num(vort_colors, nan=0.0)
            vel_colors = np.nan_to_num(vel_colors, nan=0.0)
            
            # Mezclar colores con pesos dinámicos que cambian en el tiempo
            alpha = 0.5 + 0.3 * np.sin(self.time * 0.1)  # Alpha oscilante
            colors = alpha * vort_colors + (1 - alpha) * vel_colors
            
            # Final safety check
            colors = np.nan_to_num(colors, nan=0.0)
            
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
        
        # Imagen inicial
        art_frame = self.create_art_frame(art_style)
        im = ax.imshow(art_frame, extent=[0, self.width, 0, self.height])
        
        def animate(frame):
            """Función de animación para cada frame"""
            # Avanzar simulación MÚLTIPLES pasos por frame para cambios visibles
            for _ in range(5):  # Más pasos por frame para mayor evolución
                self.step_simulation()
            
            # Actualizar arte
            art_frame = self.create_art_frame(art_style)
            im.set_array(art_frame)
            
            ax.set_title(f'Arte Procedural Dinámico - Frame {frame}\n'
                        f'Ecuaciones de Euler 2D (Tiempo: {self.time})', fontsize=12)
            
            return [im]
        
        # Crear animación con intervalo más corto para mayor fluidez
        anim = FuncAnimation(fig, animate, frames=frames, 
                           interval=80, blit=True, repeat=True)
        
        if save_path:
            print(f"Guardando animación en: {save_path}")
            anim.save(save_path, writer='pillow', fps=12)
        
        plt.show()
        return anim

# Ejemplo de uso del generador
if __name__ == "__main__":
    # Crear instancia del generador con resolución apropiada para las dimensiones
    generator = EulerArtGenerator(width=400, height=400, resolution=100)
    
    # Generar diferentes tipos de arte
    print("=== GENERADOR DE ARTE PROCEDURAL CON ECUACIONES DE EULER MEJORADO ===\n")
    
    # Arte estático - Danza de Vórtices
    print("1. Generando arte: Danza de Vórtices")
    art1 = generator.generate_static_art(
        steps=150,
        flow_type='vortex_dance',
        art_style='mixed_media',
        save_path='euler_art_vortex_dance_v2.png'
    )
    
    # Resetear para nuevo arte
    generator = EulerArtGenerator(width=400, height=400, resolution=100)

    # Arte estático - Galaxia Espiral
    print("2. Generando arte: Galaxia Espiral")
    art2 = generator.generate_static_art(
        steps=200,
        flow_type='spiral_galaxy',
        art_style='velocity_field',
        save_path='euler_art_spiral_galaxy.png'
    )
    
    # Resetear para nuevo Arte
    generator = EulerArtGenerator(width=400, height=400, resolution=100)

    # Crear animación mejorada
    print("3. Creando animación mejorada...")
    animation = generator.create_animation(
        frames=60,
        flow_type='vortex_dance',
        art_style='mixed_media',
        save_path='euler_art_animation_v2.gif'
    )
    
    print("\n=== ARTE DINÁMICO GENERADO EXITOSAMENTE ===")
