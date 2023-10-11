import pygame
import sys
import random

# Constants
WIDTH, HEIGHT = 400, 600
GROUND_HEIGHT = 50
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BIRD_COLOR = (255, 0, 0)
PIPE_COLOR = (0, 255, 0)

# Game variables
gravity = 0.25
pipe_width = 70
pipe_gap = 100
max_velocity = 10

# AI Constants
POPULATION_SIZE = 428
MUTATION_RATE = 0.1
SCORE_LIMIT = 25  # Score limit to determine when to stop the generation

# Initialize Pygame
pygame.init()

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

clock = pygame.time.Clock()
    

def clamp_velocity(generation, agent):
    """Limit the agent's vertical velocity within a range."""
    if generation.population[agent].agent_velocity < -max_velocity:
        generation.population[agent].agent_velocity = -max_velocity
    if generation.population[agent].agent_velocity > max_velocity:
        generation.population[agent].agent_velocity = max_velocity

def has_collided(agent, pipe_x, pipe_height):
    agent_y = agent.agent_position[1]
    agent_x = agent.agent_position[0]

    if agent_y < 0 or agent_y > HEIGHT:
        return True

    if (
        pipe_x < agent_x < pipe_x + pipe_width
        and not (pipe_height < agent_y < pipe_height + pipe_gap)
    ):
        return True

    return False


def reset_pipe_height():
    return random.randint(10, HEIGHT - GROUND_HEIGHT - pipe_gap)


def run_generation(generation, time_step=1, display_graphics=True):
    
    # Pipe variables
    pipe_x = WIDTH
    pipe_height = reset_pipe_height()

    if not display_graphics : print(generation.number)

    # Score tracking
    generation_score = 0

    # Main game loop
    game_over = False
    screen.fill(WHITE)

    generation.reset_population([20, HEIGHT // 2], 0)

    while not game_over:

        for i in range(time_step):

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Bird logic
            for i in range(len(generation.population)):
                if generation.population[i].active:  
                    Vy_norm = generation.population[i].agent_velocity / max_velocity  # Normalize vertical velocity
                    X_dist = abs(pipe_x - generation.population[i].agent_position[0])  # Horizontal distance to gap
                    X_dist_norm = X_dist / WIDTH  # Normalize horizontal distance

                    Y_dist = abs((pipe_height + pipe_gap // 2) - generation.population[i].agent_position[1])  # Vertical distance to gap
                    Y_dist_norm = -(Y_dist / (HEIGHT // 2)) if generation.population[i].agent_position[1] < pipe_height + pipe_gap // 2 else (Y_dist / (HEIGHT // 2))

                    # Pass inputs to the neural network
                    inputs = [Vy_norm, X_dist_norm, Y_dist_norm]
                    generation.population[i].process(inputs, gravity)
                    clamp_velocity(generation, i)
                    generation.population[i].move()

                    if has_collided(generation.population[i], pipe_x, pipe_height):
                        generation.population[i].active = False  # Deactivate the bird if it hits a wall
                    else:
                        generation.population[i].score += 1  # Increase the score while active

            # Update pipe position
            pipe_x -= 4
            if pipe_x < -pipe_width:
                pipe_x = WIDTH
                pipe_height = reset_pipe_height()
                generation_score += 1  # Increase the score when pipes pass
                for i in range(len(generation.population)):
                    if generation.population[i].active:
                        generation.population[i].goals += 1  

            # Check if all birds are deactivated or score limit is reached
            all_inactive = all(not agent.active for agent in generation.population)
            if all_inactive or generation_score >= SCORE_LIMIT:
                game_over = True
                break

        if display_graphics:
            # Draw everything
            screen.fill(WHITE)

            # Draw ground
            pygame.draw.rect(screen, BLUE, (0, HEIGHT - GROUND_HEIGHT, WIDTH, GROUND_HEIGHT))

            # Draw pipes as rectangles
            pygame.draw.rect(screen, PIPE_COLOR, (pipe_x, 0, pipe_width, pipe_height))
            pygame.draw.rect(
                screen,
                PIPE_COLOR,
                (pipe_x, pipe_height + pipe_gap, pipe_width, HEIGHT - pipe_height - pipe_gap),
            )

            for i in range(len(generation.population)):
                if generation.population[i].active:
                    pygame.draw.circle(
                        screen,
                        BIRD_COLOR,
                        (generation.population[i].agent_position[0], generation.population[i].agent_position[1]),
                        generation.population[i].radius
                    )


            # Display the generation score
            font = pygame.font.Font(None, 36)
            score_display = font.render(f"Current Score: {generation_score}, Generation {generation.number}", True, (0, 0, 0))
            screen.blit(score_display, (10, 10))

            pygame.display.update()
            clock.tick(30)


    pygame.time.wait(100)


