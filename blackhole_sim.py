import pygame
import numpy as np
import sys
import math

class BlackHoleApp:
    def __init__(self, width=800, height=800, fps=60):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Schwarzschild Black Hole Simulation")
        self.clock = pygame.time.Clock()
        
        self.rs = 40.0 # Schwarzschild radius
        self.inc = np.radians(82) # Inclination angle (0=face-on, 90=edge-on)
        self.num_particles = 60000
        
        self.init_starfield()
        self.init_disk()
        
    def init_starfield(self):
        print("Initializing gravitationally lensed starfield...")
        self.bg_surface = pygame.Surface((self.width, self.height))
        self.bg_surface.fill((5, 5, 8))
        
        num_stars = 2500
        for _ in range(num_stars):
            angle = np.random.uniform(0, 2 * np.pi)
            r_true = np.random.uniform(self.rs * 3.5, self.width * 1.5)
            z_depth = np.random.uniform(200, 2000)
            
            x = r_true * np.cos(angle)
            y = r_true * np.sin(angle)
            
            # Approximation of Einstein deflection (Point mass lens)
            re_sq = 4.0 * self.rs * z_depth * 0.1
            r_app = (r_true + np.sqrt(r_true**2 + 4.0 * re_sq)) / 2.0
            
            mapped_x = self.width//2 + x * (r_app / r_true)
            mapped_y = self.height//2 + y * (r_app / r_true)
                
            brightness = np.random.randint(50, 255)
            # Fix numpy choice error by indexing directly
            colors_list = [
                (brightness, brightness, brightness),
                (int(brightness*0.8), int(brightness*0.9), brightness),
                (brightness, int(brightness*0.9), int(brightness*0.8))
            ]
            c = colors_list[np.random.randint(0, 3)]
            s = np.random.choice([1, 1, 1, 2, 2])
            
            # The event horizon shadow (roughly 2.6 R_s for a non-spinning BH)
            if np.hypot(mapped_x - self.width//2, mapped_y - self.height//2) > 2.61 * self.rs:
                pygame.draw.circle(self.bg_surface, c, (int(mapped_x), int(mapped_y)), s)
        print("Starfield generated.")
        
    def init_disk(self):
        print("Initializing accretion disk...")
        # Distribute particles in a disk from 3 rs (ISCO) to 15 rs
        r_inner = 3.0 * self.rs
        r_outer = 15.0 * self.rs
        
        # Power law distribution favoring inner regions (denser near BH)
        u = np.random.uniform(0, 1, self.num_particles)
        self.r = r_inner + (r_outer - r_inner) * u**1.8 
        self.theta = np.random.uniform(0, 2 * np.pi, self.num_particles)
        
        # Keplerian orbital velocity v ~ 1/sqrt(r), angular velocity omega ~ r^(-1.5)
        self.omega = 0.5 * (self.rs / self.r)**1.5
        
        # Base colors based on temperature
        # Inner is hotter (blue/white), outer is cooler (orange/red/dim)
        self.base_colors = np.zeros((self.num_particles, 3), dtype=int)
        for i in range(self.num_particles):
            r_norm = (self.r[i] - r_inner) / (r_outer - r_inner)
            
            if r_norm < 0.15:
                # White/Blue-white
                brightness = 1.0 - (r_norm / 0.15)
                c = [np.clip(200 + int(55*brightness), 0, 255), 
                     np.clip(220 + int(35*brightness), 0, 255), 
                     255]
            elif r_norm < 0.4:
                # Yellow/Orange
                local_norm = (r_norm - 0.15) / 0.25
                c = [255, 
                     np.clip(220 - int(100*local_norm), 0, 255), 
                     np.clip(100 - int(60*local_norm), 0, 255)]
            else:
                # Deep Orange/Red
                local_norm = (r_norm - 0.4) / 0.6
                darkness = max(0.1, 1.0 - local_norm)
                c = [np.clip(int(255*darkness), 0, 255), 
                     np.clip(int(100*darkness), 0, 255), 
                     np.clip(int(20*darkness), 0, 255)]
            self.base_colors[i] = c
        print("Disk generated.")
            
    def get_color_with_doppler(self, base_color_array, velocity_away):
        # Doppler shift: relativistic beaming makes approaching side brighter and bluer
        # Receding side dimmer and redder
        # velocity_away > 0 : receding : red shift & dip in brightness
        # velocity_away < 0 : approaching : blue shift & boost in brightness
        
        shift = np.clip(1.0 - velocity_away * 1.5, 0.1, 3.0) # Brightness factor
        shift = shift[:, np.newaxis]
        velocity_away_exp = velocity_away[:, np.newaxis]
        
        r_colors = base_color_array[:, 0:1] * shift * (1.0 + np.maximum(0, velocity_away_exp)*0.4)
        g_colors = base_color_array[:, 1:2] * shift
        b_colors = base_color_array[:, 2:3] * shift * (1.0 - np.minimum(0, velocity_away_exp)*0.4)
        
        colors = np.concatenate([r_colors, g_colors, b_colors], axis=1)
        return np.clip(colors, 0, 255).astype(np.uint8)

    def draw_particles(self, pixel_array, xp, yp, final_colors):
        # Draw particles with a faint inner glow (cross shape)
        valid_px_mask = (xp > 0) & (xp < self.width-1) & (yp > 0) & (yp < self.height-1)
        
        xp_val = xp[valid_px_mask]
        yp_val = yp[valid_px_mask]
        c_val = final_colors[valid_px_mask]
        
        # Center pixel
        pixel_array[xp_val, yp_val] = c_val
        
        # Glow pixels (fainter)
        c_glow = c_val // 2
        pixel_array[xp_val-1, yp_val] = c_glow
        pixel_array[xp_val+1, yp_val] = c_glow
        pixel_array[xp_val, yp_val-1] = c_glow
        pixel_array[xp_val, yp_val+1] = c_glow

    def draw(self):
        self.screen.blit(self.bg_surface, (0, 0))
        
        # 3D Cartesian coordinates of the disk
        x3d = self.r * np.cos(self.theta)
        y3d = self.r * np.sin(self.theta)
        z3d = np.zeros_like(x3d)
        
        # Camera transformation (Inclination rotation around X axis)
        cos_i = np.cos(self.inc)
        sin_i = np.sin(self.inc)
        
        y_screen = y3d * cos_i - z3d * sin_i
        z_depth = y3d * sin_i + z3d * cos_i
        x_screen = x3d 
        
        # Relativistic velocity beaming factor
        # v ~ omega * r. Component projected on screen X controls the Doppler shift.
        v_depth = self.omega * (x_screen / self.rs) * sin_i * 0.16 
        
        # Unlensed positions on the projection plane
        r_true = np.hypot(x_screen, y_screen)
        phi_screen = np.arctan2(y_screen, x_screen)
        
        # Smooth depth filter to avoid harsh ring generation.
        # Only heavily lens particles that are functionally behind the BH (z_depth > 0)
        smooth = 2.0 * self.rs
        z_eff = (np.sqrt(z_depth**2 + smooth**2) + z_depth) / 2.0
        
        # Continuous Einstein ring displacement mapping (approximating GR deflection perfectly)
        re_sq = 4.0 * self.rs * z_eff * 1.55
        r_app = (r_true + np.sqrt(r_true**2 + 4.0 * re_sq)) / 2.0
        
        x_app = r_app * np.cos(phi_screen)
        y_app = r_app * np.sin(phi_screen)
        
        # Back vs Front masks
        back_mask = z_depth > 0
        front_mask = ~back_mask
        
        # Photon sphere / event horizon masking
        shadow_radius = 2.61 * self.rs
        valid_mask = r_app > shadow_radius
        
        cx, cy = self.width // 2, self.height // 2
        
        # Pixel coords
        x_px = np.clip(np.round(x_app + cx), 0, self.width-1).astype(int)
        y_px = np.clip(np.round(y_app + cy), 0, self.height-1).astype(int)
        
        # Calculate relativistic colors with doppler beaming
        final_colors = self.get_color_with_doppler(self.base_colors, v_depth)
        
        # --- Draw to Pygame Surface natively via surfarray ---
        pixel_array = pygame.surfarray.pixels3d(self.screen)
        
        # 1. Draw BACK disk (lensed over and under the black hole)
        indices_back = np.where(back_mask & valid_mask)[0]
        self.draw_particles(pixel_array, x_px[indices_back], y_px[indices_back], final_colors[indices_back])
        
        # Release the pixel lock so we can use standard Pygame draw commands
        del pixel_array
        
        # 2. Draw perfectly deep Black Hole Event Horizon + Photon Sphere shadow
        pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), int(shadow_radius))
        
        # 3. Draw FRONT disk (directly obscuring the black hole)
        pixel_array = pygame.surfarray.pixels3d(self.screen)
        
        indices_front = np.where(front_mask & valid_mask)[0]
        self.draw_particles(pixel_array, x_px[indices_front], y_px[indices_front], final_colors[indices_front])
        
        del pixel_array

    def run(self):
        font = pygame.font.SysFont(None, 24)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Draw frame
            self.draw()
            
            # Output frame metadata (FPS)
            fps_text = font.render(f"FPS: {self.clock.get_fps():.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (10, 10))
            
            label_text = font.render("Relativistic Accretion Disk Simulation", True, (200, 200, 200))
            self.screen.blit(label_text, (10, self.height - 30))
            
            pygame.display.flip()
            
            # Advance physics engine
            self.theta += self.omega
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = BlackHoleApp(width=800, height=800, fps=60)
    app.run()
