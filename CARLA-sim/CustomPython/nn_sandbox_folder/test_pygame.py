import pygame

pygame.init()
print("pygame.get_init():", pygame.get_init())
print("display.get_init():", pygame.display.get_init())

screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Pygame Test")
print("display surface after set_mode:", pygame.display.get_surface())

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
print("Pygame quit cleanly.")
