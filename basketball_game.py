import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import sys
from typing import Tuple, Optional

# Initialize Pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors - Enhanced color palette
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 140, 0)
DARK_ORANGE = (220, 100, 0)
RED = (220, 20, 20)
DARK_RED = (150, 0, 0)
GREEN = (34, 139, 34)
BRIGHT_GREEN = (50, 255, 50)
NEON_GREEN = (57, 255, 20)
BLUE = (30, 144, 255)
DEEP_BLUE = (0, 100, 200)
BROWN = (139, 69, 19)
LIGHT_BROWN = (205, 133, 63)
GRAY = (128, 128, 128)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (64, 64, 64)
COURT_COLOR = (240, 217, 181)  # Enhanced court color
WOOD_COLOR = (101, 67, 33)     # Darker wood
SKY_BLUE = (135, 206, 235)
GOLD = (255, 215, 0)
BRIGHT_GOLD = (255, 230, 50)
SILVER = (192, 192, 192)
NAVY = (25, 25, 112)
CREAM = (255, 253, 208)

# Basketball constants
BALL_RADIUS = 15
GRAVITY = 0.5
BOUNCE_DAMPING = 0.8
FRICTION = 0.98

class Particle:
    def __init__(self, x: float, y: float, color: tuple, life: int = 30):
        self.x = x
        self.y = y
        self.vx = (np.random.random() - 0.5) * 10
        self.vy = (np.random.random() - 0.5) * 10
        self.color = color
        self.life = life
        self.max_life = life
        self.size = np.random.randint(2, 6)
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2  # gravity
        self.vx *= 0.99  # friction
        self.life -= 1
        
    def draw(self, screen):
        if self.life > 0:
            alpha = self.life / self.max_life
            current_size = max(1, int(self.size * alpha))
            fade_color = (
                int(self.color[0] * alpha),
                int(self.color[1] * alpha),
                int(self.color[2] * alpha)
            )
            pygame.draw.circle(screen, fade_color, (int(self.x), int(self.y)), current_size)
    
    def is_alive(self):
        return self.life > 0

class ParticleSystem:
    def __init__(self):
        self.particles = []
    
    def add_explosion(self, x: float, y: float, color: tuple, count: int = 15):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))
    
    def update(self):
        for particle in self.particles:
            particle.update()
        self.particles = [p for p in self.particles if p.is_alive()]
    
    def draw(self, screen):
        for particle in self.particles:
            particle.draw(screen)

# Hoop constants
HOOP_X = SCREEN_WIDTH - 150
HOOP_Y = 200
HOOP_WIDTH = 80
HOOP_HEIGHT = 10
BACKBOARD_WIDTH = 10
BACKBOARD_HEIGHT = 120

class Ball:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = BALL_RADIUS
        self.color = ORANGE
        self.trail = []
        self.max_trail_length = 10
        
    def update(self):
        # Apply gravity
        self.vy += GRAVITY
        
        # Apply friction
        self.vx *= FRICTION
        self.vy *= FRICTION
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Add to trail
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
        
        # Boundary collision
        if self.x - self.radius <= 0 or self.x + self.radius >= SCREEN_WIDTH:
            self.vx = -self.vx * BOUNCE_DAMPING
            self.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.x))
            
        if self.y - self.radius <= 0:
            self.vy = -self.vy * BOUNCE_DAMPING
            self.y = self.radius
            
        if self.y + self.radius >= SCREEN_HEIGHT - 60:  # Ground bar height is 60
            self.vy = -self.vy * BOUNCE_DAMPING
            self.y = SCREEN_HEIGHT - 60 - self.radius
            if abs(self.vy) < 1:  # Stop bouncing when velocity is very small
                self.vy = 0
    
    def draw(self, screen):
        # Draw trail with enhanced gradient effect
        for i, pos in enumerate(self.trail):
            if i < len(self.trail) - 1:  # Only draw if not the last position
                alpha = (i + 1) / len(self.trail)
                trail_radius = max(2, int(self.radius * alpha * 0.8))
                
                # Multi-layer trail for glow effect
                for layer in range(3):
                    layer_alpha = alpha * (0.3 + layer * 0.2)
                    layer_radius = trail_radius + (2 - layer)
                    trail_color = (
                        int(ORANGE[0] * layer_alpha),
                        int(ORANGE[1] * layer_alpha),
                        int(ORANGE[2] * layer_alpha)
                    )
                    if layer_radius > 0:
                        pygame.draw.circle(screen, trail_color, pos, layer_radius)
        
        # Draw basketball shadow with blur effect
        shadow_offset = 4
        for blur in range(3):
            shadow_alpha = 50 - (blur * 15)
            shadow_color = (shadow_alpha, shadow_alpha, shadow_alpha)
            shadow_pos = (int(self.x + shadow_offset + blur), int(self.y + shadow_offset + blur))
            pygame.draw.circle(screen, shadow_color, shadow_pos, self.radius + blur, 0)
        
        # Draw basketball with multiple layers for 3D effect
        # Base layer (darkest)
        pygame.draw.circle(screen, DARK_ORANGE, (int(self.x), int(self.y)), self.radius)
        
        # Middle layer
        pygame.draw.circle(screen, ORANGE, (int(self.x), int(self.y)), self.radius - 1)
        
        # Top layer (lightest)
        pygame.draw.circle(screen, (255, 160, 50), (int(self.x), int(self.y)), self.radius - 3)
        
        # Basketball pattern lines with better styling
        line_width = 3
        line_color = (80, 40, 0)  # Dark brown for lines
        
        # Horizontal curved lines
        pygame.draw.arc(screen, line_color, 
                       (self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2),
                       0, math.pi, line_width)
        pygame.draw.arc(screen, line_color,
                       (self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2),
                       math.pi, 2 * math.pi, line_width)
        
        # Vertical lines
        pygame.draw.line(screen, line_color, 
                        (int(self.x), int(self.y - self.radius + 3)), 
                        (int(self.x), int(self.y + self.radius - 3)), line_width)
        
        # Curved side lines for more realistic basketball pattern
        left_arc = pygame.Rect(self.x - self.radius, self.y - self.radius//2, self.radius, self.radius)
        right_arc = pygame.Rect(self.x, self.y - self.radius//2, self.radius, self.radius)
        pygame.draw.arc(screen, line_color, left_arc, -math.pi/2, math.pi/2, 2)
        pygame.draw.arc(screen, line_color, right_arc, math.pi/2, 3*math.pi/2, 2)
        
        # Multiple highlights for enhanced 3D effect
        highlight_positions = [
            (int(self.x - self.radius * 0.4), int(self.y - self.radius * 0.4)),
            (int(self.x - self.radius * 0.2), int(self.y - self.radius * 0.6))
        ]
        highlight_colors = [(255, 220, 150), (255, 200, 100)]
        highlight_sizes = [max(4, self.radius // 3), max(2, self.radius // 5)]
        
        for i, (pos, color, size) in enumerate(zip(highlight_positions, highlight_colors, highlight_sizes)):
            pygame.draw.circle(screen, color, pos, size)

class Hoop:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.width = HOOP_WIDTH
        self.height = HOOP_HEIGHT
        self.backboard_x = x + self.width
        self.backboard_y = y - BACKBOARD_HEIGHT // 2
        self.scored = False
        
    def check_collision(self, ball: Ball) -> bool:
        # Check if ball is above the hoop and falling through it
        ball_bottom = ball.y + ball.radius
        ball_top = ball.y - ball.radius
        ball_left = ball.x - ball.radius
        ball_right = ball.x + ball.radius
        
        # Check if ball is in the hoop area and moving downward
        if (ball_left > self.x and ball_right < self.x + self.width and
            ball_top < self.y and ball_bottom > self.y and ball.vy > 0):
            return True
        return False
        
    def check_backboard_collision(self, ball: Ball):
        # Check collision with backboard
        if (ball.x + ball.radius >= self.backboard_x and
            ball.y >= self.backboard_y and ball.y <= self.backboard_y + BACKBOARD_HEIGHT):
            ball.vx = -abs(ball.vx) * BOUNCE_DAMPING
            ball.x = self.backboard_x - ball.radius
    
    def draw(self, screen):
        # Draw enhanced backboard with wood texture effect
        shadow_offset = 3
        pygame.draw.rect(screen, (40, 40, 40), 
                        (self.backboard_x + shadow_offset, self.backboard_y + shadow_offset, 
                         BACKBOARD_WIDTH, BACKBOARD_HEIGHT))
        
        # Main backboard with gradient
        pygame.draw.rect(screen, LIGHT_GRAY, 
                        (self.backboard_x, self.backboard_y, BACKBOARD_WIDTH, BACKBOARD_HEIGHT))
        
        # Backboard highlights and details
        pygame.draw.rect(screen, WHITE, 
                        (self.backboard_x + 1, self.backboard_y, BACKBOARD_WIDTH - 2, BACKBOARD_HEIGHT), 2)
        
        # Wood grain effect lines
        for i in range(0, BACKBOARD_HEIGHT, 15):
            grain_y = self.backboard_y + i
            pygame.draw.line(screen, (200, 200, 200), 
                           (self.backboard_x, grain_y), 
                           (self.backboard_x + BACKBOARD_WIDTH, grain_y), 1)
        
        # Enhanced target square with glow effect
        square_size = 45
        square_x = self.backboard_x - square_size - 8
        square_y = self.y - square_size // 2
        
        # Square glow effect
        for glow in range(3):
            glow_color = (255, 255, 255, 100 - glow * 30)
            glow_rect = (square_x - glow, square_y - glow, 
                        square_size + glow * 2, square_size + glow * 2)
            pygame.draw.rect(screen, RED, glow_rect, 2)
        
        # Main target square
        pygame.draw.rect(screen, RED, (square_x, square_y, square_size, square_size), 4)
        pygame.draw.rect(screen, WHITE, (square_x + 2, square_y + 2, square_size - 4, square_size - 4), 2)
        
        # Enhanced hoop rim with metallic 3D effect
        rim_thickness = 12
        
        # Multiple shadow layers for depth
        for shadow in range(3):
            shadow_y = self.y + shadow + 1
            shadow_color = (100 - shadow * 20, 0, 0)
            pygame.draw.ellipse(screen, shadow_color, 
                               (self.x - shadow, shadow_y, self.width + shadow * 2, rim_thickness + shadow))
        
        # Main rim with gradient effect
        rim_colors = [DARK_RED, RED, (255, 80, 80), (255, 120, 120)]
        for i, color in enumerate(rim_colors):
            rim_y = self.y + i
            rim_width = self.width - i * 2
            rim_x = self.x + i
            if rim_width > 0:
                pygame.draw.ellipse(screen, color, (rim_x, rim_y, rim_width, rim_thickness - i))
        
        # Rim shine highlights
        shine_width = self.width // 4
        pygame.draw.ellipse(screen, BRIGHT_GOLD, 
                           (self.x + shine_width, self.y + 1, shine_width, rim_thickness // 3))
        pygame.draw.ellipse(screen, WHITE, 
                           (self.x + shine_width + 5, self.y + 2, shine_width - 10, 2))
        
        # Enhanced realistic net
        net_segments = 16
        net_length = 40
        
        for i in range(net_segments):
            # Calculate position along the rim
            angle = (i / net_segments) * 2 * math.pi
            start_x = self.x + self.width/2 + (self.width/2 - 8) * math.cos(angle)
            start_y = self.y + rim_thickness
            
            # Create curved net strands
            strand_points = [(start_x, start_y)]
            
            for j in range(1, 5):  # 4 segments per strand
                segment_progress = j / 4
                curve_factor = math.sin(segment_progress * math.pi) * 8
                
                segment_y = start_y + (net_length * segment_progress)
                segment_x = start_x + math.sin(angle + segment_progress * 0.3) * curve_factor
                
                strand_points.append((segment_x, segment_y))
            
            # Draw the net strand with multiple lines for thickness
            for thickness in range(3):
                color = WHITE if thickness == 0 else (240, 240, 240)
                for k in range(len(strand_points) - 1):
                    start_point = (strand_points[k][0] + thickness - 1, strand_points[k][1])
                    end_point = (strand_points[k + 1][0] + thickness - 1, strand_points[k + 1][1])
                    pygame.draw.line(screen, color, start_point, end_point, 2)
        
        # Add cross-connecting net lines
        for i in range(0, net_segments - 1, 4):
            angle1 = (i / net_segments) * 2 * math.pi
            angle2 = ((i + 4) / net_segments) * 2 * math.pi
            
            for level in range(1, 4):
                y_level = start_y + (net_length * level / 4)
                x1 = self.x + self.width/2 + (self.width/2 - 8) * math.cos(angle1)
                x2 = self.x + self.width/2 + (self.width/2 - 8) * math.cos(angle2)
                
                curve1 = math.sin(level / 4 * math.pi) * 8
                curve2 = math.sin(level / 4 * math.pi) * 8
                
                x1 += math.sin(angle1 + level / 4 * 0.3) * curve1
                x2 += math.sin(angle2 + level / 4 * 0.3) * curve2
                
                pygame.draw.line(screen, (220, 220, 220), (x1, y_level), (x2, y_level), 1)

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Warning: Could not open camera. Using mouse control instead.")
                self.cap = None
        except Exception as e:
            print(f"Camera initialization failed: {e}. Using mouse control instead.")
            self.cap = None
        
        self.hand_pos = None
        self.prev_hand_pos = None
        self.hand_velocity = (0, 0)
        self.using_mouse = self.cap is None
        
    def update(self) -> Optional[Tuple[float, float]]:
        if self.using_mouse:
            # Use mouse as fallback
            mouse_pos = pygame.mouse.get_pos()
            if self.hand_pos is not None:
                self.hand_velocity = (mouse_pos[0] - self.hand_pos[0], mouse_pos[1] - self.hand_pos[1])
            self.hand_pos = mouse_pos
            return self.hand_pos
            
        if self.cap is None:
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                return None
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get middle finger tip (landmark 12)
                    middle_tip = hand_landmarks.landmark[12]
                    
                    # Convert to screen coordinates
                    h, w = frame.shape[:2]
                    x = int(middle_tip.x * SCREEN_WIDTH)
                    y = int(middle_tip.y * SCREEN_HEIGHT)
                    
                    # Calculate velocity
                    if self.hand_pos is not None:
                        self.hand_velocity = (x - self.hand_pos[0], y - self.hand_pos[1])
                    
                    self.prev_hand_pos = self.hand_pos
                    self.hand_pos = (x, y)
                    
                    # Draw hand landmarks on camera feed
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            else:
                self.hand_pos = None
                self.hand_velocity = (0, 0)
            
            # Show camera feed (smaller window)
            frame_small = cv2.resize(frame, (320, 240))
            cv2.imshow('Hand Tracking (Press Q to close)', frame_small)
            
            # Non-blocking key check
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.close()
            
        except Exception as e:
            print(f"Error in hand tracking: {e}")
            return None
            
        return self.hand_pos
    
    def get_velocity(self) -> Tuple[float, float]:
        return self.hand_velocity
    
    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

class BasketballGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Hand Motion Basketball Game")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Game objects
        self.ball = Ball(100, SCREEN_HEIGHT - 200)  # Start ball higher up
        self.hoop = Hoop(HOOP_X, HOOP_Y)
        self.hand_tracker = HandTracker()
        self.particles = ParticleSystem()
        
        # Ground bar properties
        self.ground_height = 60
        self.ground_color = (101, 67, 33)  # Dark brown wood color
        
        # Game state
        self.score = 0
        self.shots_taken = 0
        self.font = pygame.font.Font(None, 36)
        self.last_score_time = 0
        
        # Hand control
        self.hand_controlling = False
        self.shoot_threshold = 10  # Velocity threshold for shooting
        
        print("=== BASKETBALL GAME CONTROLS ===")
        print("1. HAND CONTROL: Move hand in front of camera")
        print("2. MOUSE CONTROL: Click to position ball")
        print("3. SPACEBAR: Shoot the ball")
        print("4. R: Reset ball position")
        print("5. ESC: Quit game")
        print("=================================")
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset ball
                    self.ball = Ball(100, SCREEN_HEIGHT - 200)  # Reset higher up from ground bar
                elif event.key == pygame.K_SPACE:
                    # Shoot with spacebar
                    self.ball.vx = 15
                    self.ball.vy = -20
                    self.shots_taken += 1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Move ball to mouse position
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.ball.x = mouse_x
                    self.ball.y = mouse_y
                    self.ball.vx = 0
                    self.ball.vy = 0
    
    def update(self):
        # Update hand tracking
        hand_pos = self.hand_tracker.update()
        
        if hand_pos:
            hand_x, hand_y = hand_pos
            hand_vx, hand_vy = self.hand_tracker.get_velocity()
            
            # Control ball with hand when it's close
            distance = math.sqrt((self.ball.x - hand_x)**2 + (self.ball.y - hand_y)**2)
            
            if distance < 50 or self.hand_controlling:
                self.hand_controlling = True
                
                # Move ball towards hand position
                if distance > 10:
                    direction_x = (hand_x - self.ball.x) / distance
                    direction_y = (hand_y - self.ball.y) / distance
                    self.ball.x += direction_x * min(distance, 10)
                    self.ball.y += direction_y * min(distance, 10)
                
                # Check for shooting gesture (quick upward movement)
                if hand_vy < -self.shoot_threshold and abs(hand_vx) < 20:
                    self.ball.vx = hand_vx * 0.3
                    self.ball.vy = hand_vy * 0.3
                    self.hand_controlling = False
                    self.shots_taken += 1
            
            # Stop controlling if hand moves too far away
            if distance > 100:
                self.hand_controlling = False
        
        # Update ball physics
        if not self.hand_controlling:
            self.ball.update()
        
        # Update particles
        self.particles.update()
        
        # Check hoop collision
        if self.hoop.check_collision(self.ball):
            if not self.hoop.scored:
                self.score += 2
                self.hoop.scored = True
                self.last_score_time = pygame.time.get_ticks()
                # Add celebration particles
                self.particles.add_explosion(self.hoop.x + self.hoop.width//2, 
                                           self.hoop.y, GOLD, 20)
                self.particles.add_explosion(self.ball.x, self.ball.y, BRIGHT_GREEN, 15)
                print(f"Score! Total: {self.score}")
        else:
            self.hoop.scored = False
        
        # Check backboard collision
        self.hoop.check_backboard_collision(self.ball)
    
    def draw(self):
        # Draw enhanced basketball court background
        # Sky gradient background
        for y in range(SCREEN_HEIGHT):
            gradient_factor = y / SCREEN_HEIGHT
            sky_r = int(135 + (255 - 135) * gradient_factor)
            sky_g = int(206 + (248 - 206) * gradient_factor)
            sky_b = int(235 + (255 - 235) * gradient_factor)
            pygame.draw.line(self.screen, (sky_r, sky_g, sky_b), (0, y), (SCREEN_WIDTH, y))
        
        # Draw wooden court flooring with perspective
        court_margin = 40
        court_rect = pygame.Rect(court_margin, court_margin + 50, 
                                SCREEN_WIDTH - 2*court_margin, SCREEN_HEIGHT - 2*court_margin - 50)
        
        # Court shadow for depth
        shadow_rect = pygame.Rect(court_margin + 5, court_margin + 55, 
                                 SCREEN_WIDTH - 2*court_margin, SCREEN_HEIGHT - 2*court_margin - 50)
        pygame.draw.rect(self.screen, (60, 40, 20), shadow_rect)
        
        # Main court
        pygame.draw.rect(self.screen, COURT_COLOR, court_rect)
        
        # Wood grain effect
        for i in range(court_margin, SCREEN_WIDTH - court_margin, 25):
            grain_color = (220, 200, 160) if (i // 25) % 2 == 0 else (210, 190, 150)
            grain_rect = pygame.Rect(i, court_margin + 50, 25, SCREEN_HEIGHT - 2*court_margin - 50)
            pygame.draw.rect(self.screen, grain_color, grain_rect)
            # Add wood grain lines
            for j in range(court_margin + 50, SCREEN_HEIGHT - court_margin, 40):
                pygame.draw.line(self.screen, (200, 180, 140), (i, j), (i + 25, j), 1)
        
        # Court border with multiple lines
        pygame.draw.rect(self.screen, WHITE, court_rect, 5)
        pygame.draw.rect(self.screen, NAVY, court_rect, 3)
        pygame.draw.rect(self.screen, WHITE, court_rect, 1)
        
        # Enhanced court markings
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        # Center line with style
        pygame.draw.line(self.screen, WHITE, 
                        (center_x, court_margin + 50), (center_x, SCREEN_HEIGHT - court_margin), 5)
        pygame.draw.line(self.screen, NAVY, 
                        (center_x - 1, court_margin + 50), (center_x - 1, SCREEN_HEIGHT - court_margin), 1)
        pygame.draw.line(self.screen, NAVY, 
                        (center_x + 1, court_margin + 50), (center_x + 1, SCREEN_HEIGHT - court_margin), 1)
        
        # Center circle with enhanced design
        pygame.draw.circle(self.screen, WHITE, (center_x, center_y), 80, 5)
        pygame.draw.circle(self.screen, NAVY, (center_x, center_y), 80, 2)
        pygame.draw.circle(self.screen, WHITE, (center_x, center_y), 25, 3)
        
        # Three-point line (enhanced arc)
        three_point_rect = pygame.Rect(HOOP_X - 220, HOOP_Y - 120, 220, 240)
        pygame.draw.arc(self.screen, WHITE, three_point_rect, 
                       math.pi/2, 3*math.pi/2, 5)
        pygame.draw.arc(self.screen, NAVY, three_point_rect, 
                       math.pi/2, 3*math.pi/2, 2)
        
        # Free throw line and circle
        ft_line_x = HOOP_X - 180
        pygame.draw.line(self.screen, WHITE,
                        (ft_line_x, HOOP_Y - 90), (ft_line_x, HOOP_Y + 90), 5)
        pygame.draw.line(self.screen, NAVY,
                        (ft_line_x - 1, HOOP_Y - 90), (ft_line_x - 1, HOOP_Y + 90), 1)
        
        # Free throw circle
        pygame.draw.circle(self.screen, WHITE, (ft_line_x, HOOP_Y), 60, 5)
        pygame.draw.circle(self.screen, NAVY, (ft_line_x, HOOP_Y), 60, 2)
        
        # Key/paint area
        key_rect = pygame.Rect(ft_line_x, HOOP_Y - 90, HOOP_X - ft_line_x, 180)
        pygame.draw.rect(self.screen, (240, 220, 190), key_rect)
        pygame.draw.rect(self.screen, WHITE, key_rect, 5)
        pygame.draw.rect(self.screen, NAVY, key_rect, 2)
        
        # Draw ground bar at the bottom
        ground_rect = pygame.Rect(0, SCREEN_HEIGHT - self.ground_height, SCREEN_WIDTH, self.ground_height)
        
        # Ground shadow for depth
        shadow_rect = pygame.Rect(0, SCREEN_HEIGHT - self.ground_height - 3, SCREEN_WIDTH, self.ground_height)
        pygame.draw.rect(self.screen, (60, 40, 20), shadow_rect)
        
        # Main ground bar with wood texture
        pygame.draw.rect(self.screen, self.ground_color, ground_rect)
        
        # Wood grain texture for ground
        for i in range(0, SCREEN_WIDTH, 30):
            grain_color = (120, 80, 40) if (i // 30) % 2 == 0 else (110, 70, 35)
            grain_rect = pygame.Rect(i, SCREEN_HEIGHT - self.ground_height, 30, self.ground_height)
            pygame.draw.rect(self.screen, grain_color, grain_rect)
            # Wood grain lines
            for j in range(SCREEN_HEIGHT - self.ground_height + 10, SCREEN_HEIGHT, 15):
                pygame.draw.line(self.screen, (140, 90, 50), (i, j), (i + 30, j), 1)
        
        # Ground border with highlight
        pygame.draw.line(self.screen, (160, 120, 80), (0, SCREEN_HEIGHT - self.ground_height), 
                        (SCREEN_WIDTH, SCREEN_HEIGHT - self.ground_height), 3)
        pygame.draw.line(self.screen, (80, 50, 25), (0, SCREEN_HEIGHT - self.ground_height + 1), 
                        (SCREEN_WIDTH, SCREEN_HEIGHT - self.ground_height + 1), 1)
        
        # Draw game objects
        self.ball.draw(self.screen)
        self.hoop.draw(self.screen)
        
        # Draw particles
        self.particles.draw(self.screen)
        
        # Enhanced UI with modern design
        # Main score panel (top-left) - Compact and prominent
        main_panel = pygame.Rect(20, 20, 260, 100)
        pygame.draw.rect(self.screen, (0, 0, 0, 220), main_panel)
        pygame.draw.rect(self.screen, BRIGHT_GOLD, main_panel, 3)
        pygame.draw.rect(self.screen, WHITE, pygame.Rect(22, 22, 256, 96), 1)
        
        # Score display with enhanced styling
        font_huge = pygame.font.Font(None, 52)
        font_large = pygame.font.Font(None, 38)
        font_medium = pygame.font.Font(None, 26)
        font_small = pygame.font.Font(None, 18)
        font_tiny = pygame.font.Font(None, 16)
        
        # Score with glow effect
        score_text = font_huge.render(f"{self.score}", True, BRIGHT_GOLD)
        score_label = font_medium.render("SCORE", True, WHITE)
        
        # Add subtle glow to score
        for offset in [(1, 1), (-1, -1)]:
            glow_text = font_huge.render(f"{self.score}", True, (150, 100, 0))
            self.screen.blit(glow_text, (30 + offset[0], 35 + offset[1]))
        
        self.screen.blit(score_text, (30, 35))
        self.screen.blit(score_label, (30, 20))
        
        # Stats in compact layout
        shots_text = font_medium.render(f"Shots: {self.shots_taken}", True, WHITE)
        self.screen.blit(shots_text, (30, 75))
        
        if self.shots_taken > 0:
            accuracy = (self.score / 2) / self.shots_taken * 100
            accuracy_color = NEON_GREEN if accuracy >= 70 else BRIGHT_GOLD if accuracy >= 40 else RED
            accuracy_text = font_medium.render(f"Accuracy: {accuracy:.1f}%", True, accuracy_color)
            self.screen.blit(accuracy_text, (150, 75))
        
        # SMALL CONTROL PANEL - Top-right corner as shown in screenshot
        control_panel = pygame.Rect(SCREEN_WIDTH - 220, 20, 200, 100)
        pygame.draw.rect(self.screen, (10, 10, 20, 180), control_panel)
        pygame.draw.rect(self.screen, SILVER, control_panel, 2)
        
        # Minimal control instructions
        control_title = font_small.render("Controls", True, WHITE)
        self.screen.blit(control_title, (SCREEN_WIDTH - 215, 25))
        
        mini_controls = [
            "Click: Position ball",
            "Space: Shoot",
            "R: Reset | Esc: Quit"
        ]
        
        for i, text in enumerate(mini_controls):
            color = NEON_GREEN if i == 1 else WHITE  # Highlight shoot control
            rendered_text = font_tiny.render(text, True, color)
            self.screen.blit(rendered_text, (SCREEN_WIDTH - 215, 45 + i * 16))
        
        # Tiny status indicator
        mode_text = "Mouse" if self.hand_tracker.using_mouse else "Camera"
        mode_color = BLUE if self.hand_tracker.using_mouse else NEON_GREEN
        status_surface = font_tiny.render(f"Mode: {mode_text}", True, mode_color)
        self.screen.blit(status_surface, (SCREEN_WIDTH - 215, 105))
        
        # Subtle hand tracking indicator - minimal and non-disruptive
        hand_pos = self.hand_tracker.hand_pos
        if hand_pos:
            # Only show a very small, subtle indicator
            current_time = pygame.time.get_ticks()
            ring_size = 8 + math.sin(current_time * 0.005) * 2  # Much smaller and slower animation
            
            # Check if hand indicator would overlap with UI - if so, make it even smaller
            in_ui_area = (hand_pos[0] < 300 and hand_pos[1] < 140) or \
                        (hand_pos[0] > SCREEN_WIDTH - 240 and hand_pos[1] < 140)
            
            if in_ui_area:
                ring_size = ring_size * 0.5  # Even smaller when near UI
            
            # Very subtle glow - only 2 rings
            for radius in range(int(ring_size + 4), int(ring_size), -2):
                alpha_factor = (ring_size + 4 - radius) / 4
                glow_intensity = int(120 * alpha_factor * 0.3)  # Much more subtle
                
                if self.hand_tracker.using_mouse:
                    glow_color = (0, glow_intensity//2, glow_intensity)
                else:
                    glow_color = (0, glow_intensity, glow_intensity//2)
                
                pygame.draw.circle(self.screen, glow_color, hand_pos, radius, 1)
            
            # Very small main indicator
            main_color = BLUE if self.hand_tracker.using_mouse else NEON_GREEN
            pygame.draw.circle(self.screen, main_color, hand_pos, int(ring_size), 2)
            pygame.draw.circle(self.screen, WHITE, hand_pos, 3)
            
            # No label when near UI areas to avoid clutter
            if not in_ui_area:
                # Tiny label only when away from UI
                control_text = "M" if self.hand_tracker.using_mouse else "C"  # Single letter
                control_color = BLUE if self.hand_tracker.using_mouse else NEON_GREEN
                
                label_surface = font_tiny.render(control_text, True, control_color)
                label_x = hand_pos[0] - 3
                label_y = hand_pos[1] - 18
                
                # Tiny background
                bg_rect = pygame.Rect(label_x - 2, label_y - 1, 8, 10)
                pygame.draw.rect(self.screen, (0, 0, 0, 120), bg_rect)
                self.screen.blit(label_surface, (label_x, label_y))
        
        # Minimal performance indicator - top-right corner, very small
        fps_text = font_tiny.render(f"{int(self.clock.get_fps())}", True, (150, 150, 150))
        self.screen.blit(fps_text, (SCREEN_WIDTH - 25, 5))
        
        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        self.hand_tracker.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = BasketballGame()
    game.run()
