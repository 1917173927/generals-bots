import math
from typing import Any, TypeAlias

import numpy as np
import pygame

from generals.core.config import Dimension, Path
from generals.gui.properties import GuiMode, Properties

Color: TypeAlias = tuple[int, int, int]
FOG_OF_WAR: Color = (62, 67, 72)
NEUTRAL_CASTLE: Color = (150, 154, 160)
VISIBLE_PATH: Color = (232, 235, 239)
VISIBLE_MOUNTAIN: Color = (184, 188, 194)
BLACK: Color = (0, 0, 0)
WHITE: Color = (230, 230, 230)
SELECTED_CELL: Color = (255, 214, 64)
VALID_TARGET: Color = (46, 204, 113)
AI_PREVIEW_PRIMARY: Color = (21, 101, 216)
AI_PREVIEW_SECONDARY: Color = (90, 150, 255)
AI_PREVIEW_TEXT: Color = (32, 36, 42)
AI_PREVIEW_MUTED: Color = (112, 118, 128)
PANEL_BG: Color = (244, 246, 249)
PANEL_SURFACE: Color = (255, 255, 255)
PANEL_BORDER: Color = (204, 211, 220)
PANEL_TEXT: Color = (31, 35, 40)
PANEL_MUTED: Color = (99, 108, 118)
PANEL_HEADER: Color = (235, 239, 244)
GRID_LINE: Color = (116, 124, 134)
ARMY_DARK: Color = (28, 32, 36)
ARMY_LIGHT: Color = (255, 255, 255)


def get_policy_candidate_arrow(candidate: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return source/target cells for a non-pass policy candidate."""
    if getattr(candidate, "is_pass", False):
        return None
    source = getattr(candidate, "source", None)
    target = getattr(candidate, "target", None)
    if source is None or target is None:
        return None
    return source, target


def format_policy_preview_lines(preview: Any, max_candidates: int = 5) -> list[str]:
    """Format a compact policy preview for the right panel."""
    if preview is None:
        return ["AI Preview", "No policy preview"]

    lines = ["AI Preview"]
    candidates = list(getattr(preview, "candidates", ()))[:max_candidates]
    if not candidates:
        lines.append("No candidates")
    for rank, candidate in enumerate(candidates, start=1):
        probability = getattr(candidate, "probability", 0.0)
        if getattr(candidate, "is_pass", False):
            lines.append(f"{rank}. Pass {probability:.0%}")
            continue

        source = getattr(candidate, "source", None)
        target = getattr(candidate, "target", None)
        direction = getattr(candidate, "direction_label", "Move")
        split = " split" if getattr(candidate, "is_split", False) else ""
        lines.append(f"{rank}. {source}->{target} {direction}{split} {probability:.0%}")

    value = getattr(preview, "value", None)
    if value is not None:
        lines.append(f"Value: {value:+.2f}")
    if getattr(preview, "policy_mode", "greedy") == "sample":
        lines.append("Sample mode: action is sampled")
    return lines


def get_valid_target_cells(selected_cell: tuple[int, int] | None, mountains: np.ndarray) -> list[tuple[int, int]]:
    """Return adjacent in-bounds, non-mountain targets for a selected source cell."""
    if selected_cell is None:
        return []

    row, col = selected_cell
    height, width = mountains.shape
    targets = []
    for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        target_row = row + row_delta
        target_col = col + col_delta
        if 0 <= target_row < height and 0 <= target_col < width and not bool(mountains[target_row, target_col]):
            targets.append((target_row, target_col))
    return targets


class Renderer:
    def __init__(self, properties: Properties):
        """
        Initialize the pygame GUI
        """
        pygame.init()
        pygame.display.set_caption("Generals")
        pygame.key.set_repeat(500, 64)

        self.properties = properties

        self.mode = self.properties.mode
        self.game = self.properties.game

        self.agent_data = self.properties.agent_data
        self.agent_fov = self.properties.agent_fov

        self.grid_height = self.properties.grid_height
        self.grid_width = self.properties.grid_width
        self.display_grid_width = self.properties.display_grid_width
        self.display_grid_height = self.properties.display_grid_height
        self.right_panel_width = self.properties.right_panel_width

        ############
        # Surfaces #
        ############
        window_width = self.display_grid_width + self.right_panel_width
        window_height = self.display_grid_height + 1

        width = Dimension.GUI_CELL_WIDTH.value
        height = Dimension.GUI_CELL_HEIGHT.value

        # Main window
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # Scoreboard
        self.right_panel = pygame.Surface((self.right_panel_width, window_height))
        self.score_cols = {}
        for col in ["Player", "Army", "Land"]:
            size = (width, height)
            if col == "Player":
                size = (2 * width, height)
            self.score_cols[col] = [pygame.Surface(size) for _ in range(3)]

        self.info_panel = {
            "time": pygame.Surface((self.right_panel_width // 2, height)),
            "speed": pygame.Surface((self.right_panel_width // 2, height)),
        }
        self.game_status_panel = pygame.Surface((self.right_panel_width, height))
        # Game area and tiles
        self.game_area = pygame.Surface((self.display_grid_width, self.display_grid_height))
        self.tiles = [
            [pygame.Surface((Dimension.SQUARE_SIZE.value, Dimension.SQUARE_SIZE.value)) for _ in range(self.grid_width)]
            for _ in range(self.grid_height)
        ]

        # Load pre-scaled images (crownie, citie, mountainie are already the right size)
        self._mountain_img = pygame.image.load(str(Path.MOUNTAIN_PATH), "png").convert_alpha()
        self._general_img = pygame.image.load(str(Path.GENERAL_PATH), "png").convert_alpha()
        self._city_img = pygame.image.load(Path.CITY_PATH, "png").convert_alpha()

        self._font = pygame.font.Font(Path.FONT_PATH, self.properties.font_size)
        self._debug_font = pygame.font.Font(Path.FONT_PATH, 10)  # Smaller font for debug
        self._panel_font = self._load_panel_font(14)
        self._panel_title_font = self._load_panel_font(15)
        self._preview_font = self._load_panel_font(12)

    def _load_panel_font(self, size: int) -> pygame.font.Font:
        """Prefer fonts that can render Chinese UI text, then fall back to bundled font."""
        for font_name in ("PingFang SC", "Heiti SC", "Noto Sans CJK SC", "Microsoft YaHei", "Arial Unicode MS"):
            font_path = pygame.font.match_font(font_name)
            if font_path:
                return pygame.font.Font(font_path, size)
        return pygame.font.Font(Path.FONT_PATH, size)

    def render(self, fps=None):
        self.render_grid()
        self.render_stats()
        pygame.display.flip()
        if fps:
            self.properties.clock.tick(fps)

    def render_cell_text(
        self,
        cell: pygame.Surface,
        text: str,
        fg_color: Color = BLACK,
        bg_color: Color = WHITE,
    ):
        """
        Draw a text in the middle of the cell with given foreground and background colors

        Args:
            cell: cell to draw
            text: text to write on the cell
            fg_color: foreground color of the text
            bg_color: background color of the cell
        """
        center = (cell.get_width() // 2, cell.get_height() // 2)

        text_surface = self._font.render(text, True, fg_color)
        if bg_color:
            cell.fill(bg_color)
        cell.blit(text_surface, text_surface.get_rect(center=center))

    def render_stats(self):
        """
        Draw player stats and additional info on the right panel
        """
        self.right_panel.fill(PANEL_BG)
        names = self.game.agents
        player_stats = self.game.get_infos()
        gui_cell_height = Dimension.GUI_CELL_HEIGHT.value

        info_text = {
            "time": f"Time: {str(self.game.time // 2) + ('.' if self.game.time % 2 == 1 else '')}",
            "speed": "Paused"
            if self.mode == GuiMode.REPLAY and self.properties.paused
            else f"Speed: {str(self.properties.game_speed)}x",
        }

        self._draw_score_header()
        for index, name in enumerate(names):
            self._draw_player_stat_row(index, name, player_stats[name])

        self._draw_info_strip(3 * gui_cell_height, info_text)

        if self.mode == GuiMode.GAME:
            self.render_game_status(gui_cell_height)
            self.render_policy_preview_panel(5 * gui_cell_height)
        # Render right_panel on the screen
        self.screen.blit(self.right_panel, (self.display_grid_width, 0))

    def _draw_score_header(self) -> None:
        rect = pygame.Rect(0, 0, self.right_panel_width, Dimension.GUI_CELL_HEIGHT.value)
        pygame.draw.rect(self.right_panel, PANEL_HEADER, rect)
        pygame.draw.line(self.right_panel, PANEL_BORDER, rect.bottomleft, rect.bottomright, 1)
        self._draw_panel_text("Player", (10, 7), self._preview_font, PANEL_MUTED, max_width=112)
        self._draw_panel_text("Army", (170, 7), self._preview_font, PANEL_MUTED, max_width=48, align="right")
        self._draw_panel_text("Land", (230, 7), self._preview_font, PANEL_MUTED, max_width=42, align="right")

    def _draw_player_stat_row(self, index: int, name: str, stats: dict[str, int]) -> None:
        row_top = (index + 1) * Dimension.GUI_CELL_HEIGHT.value
        rect = pygame.Rect(0, row_top, self.right_panel_width, Dimension.GUI_CELL_HEIGHT.value)
        base_color = self.agent_data.get(name, {}).get("color", PANEL_MUTED)
        active = self.agent_fov.get(name, True)
        color = base_color if active else self._blend_color(base_color, PANEL_BG, 0.55)

        pygame.draw.rect(self.right_panel, PANEL_SURFACE, rect)
        pygame.draw.line(self.right_panel, PANEL_BORDER, rect.bottomleft, rect.bottomright, 1)
        pygame.draw.rect(self.right_panel, color, pygame.Rect(10, row_top + 8, 14, 14), border_radius=3)

        name_color = PANEL_TEXT if active else PANEL_MUTED
        self._draw_panel_text(name, (32, row_top + 7), self._panel_font, name_color, max_width=116)
        self._draw_panel_text(
            str(stats["army"]),
            (170, row_top + 7),
            self._panel_font,
            PANEL_TEXT,
            max_width=48,
            align="right",
        )
        self._draw_panel_text(
            str(stats["land"]),
            (230, row_top + 7),
            self._panel_font,
            PANEL_TEXT,
            max_width=42,
            align="right",
        )

    def _draw_info_strip(self, top: int, info_text: dict[str, str]) -> None:
        rect = pygame.Rect(0, top, self.right_panel_width, Dimension.GUI_CELL_HEIGHT.value)
        pygame.draw.rect(self.right_panel, PANEL_BG, rect)
        half_width = self.right_panel_width // 2
        for index, key in enumerate(["time", "speed"]):
            item_rect = pygame.Rect(index * half_width + 6, top + 4, half_width - 12, rect.height - 8)
            pygame.draw.rect(self.right_panel, PANEL_SURFACE, item_rect, border_radius=6)
            pygame.draw.rect(self.right_panel, PANEL_BORDER, item_rect, 1, border_radius=6)
            self._draw_panel_text(
                info_text[key],
                (item_rect.x + 8, item_rect.y + 4),
                self._preview_font,
                PANEL_TEXT,
                max_width=item_rect.width - 16,
            )

    def render_game_status(self, gui_cell_height: int):
        """Draw compact playable-mode interaction status."""
        split_state = "On" if self.properties.split_enabled else "Off"
        selected_cell = self.properties.selected_cell
        selected_text = str(selected_cell) if selected_cell is not None else "-"
        text = f"{self.properties.last_game_message} | Sel {selected_text} | Split {split_state}"

        self.game_status_panel.fill(PANEL_SURFACE)
        pygame.draw.line(
            self.game_status_panel,
            PANEL_BORDER,
            (0, self.game_status_panel.get_height() - 1),
            (self.game_status_panel.get_width(), self.game_status_panel.get_height() - 1),
            1,
        )
        status_color = SELECTED_CELL if selected_cell is not None else VALID_TARGET
        pygame.draw.rect(self.game_status_panel, status_color, pygame.Rect(8, 8, 14, 14), border_radius=7)
        self._draw_surface_text(
            self.game_status_panel,
            text,
            (30, 7),
            self._preview_font,
            PANEL_TEXT,
            max_width=self.game_status_panel.get_width() - 38,
        )
        self.right_panel.blit(self.game_status_panel, (0, 4 * gui_cell_height))

    def render_policy_preview_panel(self, top: int):
        """Draw the compact Top-K policy preview panel."""
        panel_height = self.right_panel.get_height() - top
        if panel_height <= 0:
            return

        rect = pygame.Rect(0, top, self.right_panel_width, panel_height)
        pygame.draw.rect(self.right_panel, PANEL_BG, rect)
        pygame.draw.line(self.right_panel, PANEL_BORDER, rect.topleft, rect.topright, 1)

        preview = self.properties.policy_preview
        padding = 8
        y = rect.top + padding
        self._draw_panel_text(
            "AI Preview",
            (rect.left + padding, y),
            self._panel_title_font,
            PANEL_TEXT,
            max_width=rect.width - 2 * padding,
        )
        y += 23

        if preview is None:
            self._draw_panel_text(
                "No policy preview",
                (rect.left + padding, y),
                self._preview_font,
                PANEL_MUTED,
                max_width=rect.width - 2 * padding,
            )
            return

        candidates = list(getattr(preview, "candidates", ()))[:5]
        if not candidates:
            self._draw_panel_text(
                "No candidates",
                (rect.left + padding, y),
                self._preview_font,
                PANEL_MUTED,
                max_width=rect.width - 2 * padding,
            )
            return

        for rank, candidate in enumerate(candidates, start=1):
            row_height = 35
            if y + row_height > rect.bottom - padding - 34:
                break
            row_rect = pygame.Rect(rect.left + padding, y, rect.width - 2 * padding, row_height - 4)
            pygame.draw.rect(self.right_panel, PANEL_SURFACE, row_rect, border_radius=6)
            pygame.draw.rect(self.right_panel, PANEL_BORDER, row_rect, 1, border_radius=6)
            color = AI_PREVIEW_PRIMARY if rank == 1 else AI_PREVIEW_SECONDARY
            pygame.draw.circle(self.right_panel, color, (row_rect.left + 15, row_rect.centery), 10)
            rank_surface = self._preview_font.render(str(rank), True, ARMY_LIGHT)
            self.right_panel.blit(rank_surface, rank_surface.get_rect(center=(row_rect.left + 15, row_rect.centery)))

            text = self._format_policy_candidate_line(rank, candidate)
            self._draw_panel_text(
                text,
                (row_rect.left + 32, row_rect.top + 7),
                self._preview_font,
                PANEL_TEXT,
                max_width=row_rect.width - 40,
            )
            y += row_height

        footer_lines: list[str] = []
        value = getattr(preview, "value", None)
        if value is not None:
            footer_lines.append(f"Value {value:+.2f}")
        if getattr(preview, "policy_mode", "greedy") == "sample":
            footer_lines.append("Sample mode")
        if footer_lines:
            self._draw_panel_text(
                " | ".join(footer_lines),
                (rect.left + padding, rect.bottom - 24),
                self._preview_font,
                PANEL_MUTED,
                max_width=rect.width - 2 * padding,
            )

    def _format_policy_candidate_line(self, rank: int, candidate: Any) -> str:
        probability = getattr(candidate, "probability", 0.0)
        if getattr(candidate, "is_pass", False):
            return f"Pass {probability:.0%}"

        source = getattr(candidate, "source", None)
        target = getattr(candidate, "target", None)
        direction = getattr(candidate, "direction_label", "Move")
        split = " split" if getattr(candidate, "is_split", False) else ""
        return f"{source}->{target} {direction}{split} {probability:.0%}"

    def _draw_panel_text(
        self,
        text: str,
        pos: tuple[int, int],
        font: pygame.font.Font,
        color: Color,
        max_width: int | None = None,
        align: str = "left",
    ) -> pygame.Rect:
        return self._draw_surface_text(self.right_panel, text, pos, font, color, max_width=max_width, align=align)

    def _draw_surface_text(
        self,
        surface: pygame.Surface,
        text: str,
        pos: tuple[int, int],
        font: pygame.font.Font,
        color: Color,
        max_width: int | None = None,
        align: str = "left",
    ) -> pygame.Rect:
        if max_width is not None:
            text = self._truncate_text(text, font, max_width)
        text_surface = font.render(text, True, color)
        x, y = pos
        if align == "right" and max_width is not None:
            x = x + max_width - text_surface.get_width()
        surface.blit(text_surface, (x, y))
        return text_surface.get_rect(topleft=(x, y))

    def _truncate_text(self, text: str, font: pygame.font.Font, max_width: int) -> str:
        """Trim text to fit one panel line."""
        text_surface = font.render(text, True, BLACK)
        if text_surface.get_width() <= max_width:
            return text
        suffix = "..."
        while text and font.render(text + suffix, True, BLACK).get_width() > max_width:
            text = text[:-1]
        return text + suffix if text else suffix

    def render_grid(self):
        """
        Render the game grid
        """
        agents = self.game.agents
        # Maps of all owned and visible cells
        owned_map = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        visible_map = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        for agent in agents:
            ownership = self.game.channels.ownership[agent]
            owned_map = np.logical_or(owned_map, ownership)
            if self.agent_fov[agent]:
                visibility = self.game.channels.get_visibility(agent)
                visible_map = np.logical_or(visible_map, visibility)

        # Helper maps for not owned and invisible cells
        not_owned_map = np.logical_not(owned_map)
        invisible_map = np.logical_not(visible_map)

        # Draw background of visible owned squares
        for agent in agents:
            ownership = self.game.channels.ownership[agent]
            visible_ownership = np.logical_and(ownership, visible_map)
            self.draw_channel(visible_ownership, self.agent_data[agent]["color"])

        # Draw visible generals
        visible_generals = np.logical_and(self.game.channels.generals, visible_map)
        self.draw_images(visible_generals, self._general_img)

        # Draw background of visible but not owned squares
        visible_not_owned = np.logical_and(visible_map, not_owned_map)
        self.draw_channel(visible_not_owned, WHITE)

        # Draw background of squares in fog of war
        self.draw_channel(invisible_map, FOG_OF_WAR)

        # Draw background of visible mountains
        visible_mountain = np.logical_and(self.game.channels.mountains, visible_map)
        self.draw_channel(visible_mountain, VISIBLE_MOUNTAIN)

        # Draw mountains (even if they are not visible)
        self.draw_images(self.game.channels.mountains, self._mountain_img)

        # Draw background of visible neutral cities
        visible_cities = np.logical_and(self.game.channels.cities, visible_map)
        visible_cities_neutral = np.logical_and(visible_cities, self.game.channels.ownership_neutral)
        self.draw_channel(visible_cities_neutral, NEUTRAL_CASTLE)

        # Draw invisible cities as mountains
        invisible_cities = np.logical_and(self.game.channels.cities, invisible_map)
        self.draw_images(invisible_cities, self._mountain_img)

        # Draw visible cities
        self.draw_images(visible_cities, self._city_img)

        # Draw nonzero army counts on visible squares
        visible_army = self.game.channels.armies * visible_map
        visible_army_indices = self.channel_to_indices(visible_army)
        for i, j in visible_army_indices:
            self.render_cell_text(
                self.tiles[i][j],
                str(int(visible_army[i, j])),
                fg_color=self._army_text_color(i, j, owned_map),
                bg_color=None,  # Transparent background
            )

        # Draw tile type debug labels if enabled
        if self.properties.show_tile_types:
            self.draw_tile_types()

        self.draw_game_selection_highlights()

        # Blit tiles to the self.game_area
        square_size = Dimension.SQUARE_SIZE.value
        for i, j in np.ndindex(self.grid_height, self.grid_width):
            self.game_area.blit(self.tiles[i][j], (j * square_size, i * square_size))
        self.draw_grid_lines()
        self.draw_policy_preview_overlay()
        self.screen.blit(self.game_area, (0, 0))

    def channel_to_indices(self, channel: np.ndarray) -> np.ndarray:
        """
        Returns a list of indices of cells with non-zero values from specified a channel.
        """
        return np.argwhere(channel != 0)

    def draw_channel(self, channel: np.ndarray, color: Color):
        """
        Draw background and borders (left and top) for grid tiles of a given channel
        """
        for i, j in self.channel_to_indices(channel):
            self.tiles[i][j].fill(color)

    def draw_images(self, channel: np.ndarray, image: pygame.Surface):
        """
        Draw images on grid tiles of a given channel
        """
        square_size = Dimension.SQUARE_SIZE.value
        # Center the image in the cell
        img_width, img_height = image.get_size()
        x_offset = (square_size - img_width) // 2
        y_offset = (square_size - img_height) // 2
        for i, j in self.channel_to_indices(channel):
            self.tiles[i][j].blit(image, (x_offset, y_offset))

    def draw_game_selection_highlights(self):
        """Draw selected source and legal targets in playable mode."""
        if self.mode != GuiMode.GAME:
            return

        selected_cell = self.properties.selected_cell
        if selected_cell is None:
            return

        square_size = Dimension.SQUARE_SIZE.value
        selected_row, selected_col = selected_cell
        if 0 <= selected_row < self.grid_height and 0 <= selected_col < self.grid_width:
            rect = (1, 1, square_size - 2, square_size - 2)
            self._tint_tile(selected_row, selected_col, (*SELECTED_CELL, 58))
            pygame.draw.rect(self.tiles[selected_row][selected_col], SELECTED_CELL, rect, 4)

        for target_row, target_col in get_valid_target_cells(selected_cell, self.game.channels.mountains):
            rect = (3, 3, square_size - 6, square_size - 6)
            self._tint_tile(target_row, target_col, (*VALID_TARGET, 46))
            pygame.draw.rect(self.tiles[target_row][target_col], VALID_TARGET, rect, 3)

    def draw_policy_preview_overlay(self):
        """Draw Top-K policy candidates on the game board."""
        preview = self.properties.policy_preview
        if preview is None:
            return

        square_size = Dimension.SQUARE_SIZE.value
        candidates = list(getattr(preview, "candidates", ()))[:5]
        for rank, candidate in enumerate(candidates):
            arrow = get_policy_candidate_arrow(candidate)
            if arrow is None:
                continue

            source, target = arrow
            source_row, source_col = source
            target_row, target_col = target
            if not (
                0 <= source_row < self.grid_height
                and 0 <= source_col < self.grid_width
                and 0 <= target_row < self.grid_height
                and 0 <= target_col < self.grid_width
            ):
                continue

            color = AI_PREVIEW_PRIMARY if rank == 0 else AI_PREVIEW_SECONDARY
            width = 5 if rank == 0 else 3
            source_rect = pygame.Rect(
                source_col * square_size + 3,
                source_row * square_size + 3,
                square_size - 6,
                square_size - 6,
            )
            target_rect = pygame.Rect(
                target_col * square_size + 6,
                target_row * square_size + 6,
                square_size - 12,
                square_size - 12,
            )
            pygame.draw.rect(self.game_area, color, source_rect, width)
            pygame.draw.rect(self.game_area, color, target_rect, max(2, width - 1))
            self._draw_arrow(source, target, color, width)
            self._draw_policy_badge(rank + 1, getattr(candidate, "probability", 0.0), target, color)

    def _draw_arrow(self, source: tuple[int, int], target: tuple[int, int], color: Color, width: int) -> None:
        square_size = Dimension.SQUARE_SIZE.value
        source_row, source_col = source
        target_row, target_col = target
        start = (source_col * square_size + square_size // 2, source_row * square_size + square_size // 2)
        end = (target_col * square_size + square_size // 2, target_row * square_size + square_size // 2)
        pygame.draw.line(self.game_area, color, start, end, width)

        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        head_length = 12
        head_angle = math.pi / 6
        left = (
            end[0] - head_length * math.cos(angle - head_angle),
            end[1] - head_length * math.sin(angle - head_angle),
        )
        right = (
            end[0] - head_length * math.cos(angle + head_angle),
            end[1] - head_length * math.sin(angle + head_angle),
        )
        pygame.draw.polygon(self.game_area, color, [end, left, right])

    def draw_grid_lines(self) -> None:
        """Draw a consistent grid over the composited board."""
        square_size = Dimension.SQUARE_SIZE.value
        for col in range(self.grid_width + 1):
            x = col * square_size
            pygame.draw.line(self.game_area, GRID_LINE, (x, 0), (x, self.display_grid_height), 1)
        for row in range(self.grid_height + 1):
            y = row * square_size
            pygame.draw.line(self.game_area, GRID_LINE, (0, y), (self.display_grid_width, y), 1)

    def _tint_tile(self, row: int, col: int, color: tuple[int, int, int, int]) -> None:
        overlay = pygame.Surface((Dimension.SQUARE_SIZE.value, Dimension.SQUARE_SIZE.value), pygame.SRCALPHA)
        overlay.fill(color)
        self.tiles[row][col].blit(overlay, (0, 0))

    def _army_text_color(self, row: int, col: int, owned_map: np.ndarray) -> Color:
        if bool(owned_map[row, col]):
            return ARMY_LIGHT
        return ARMY_DARK

    def _draw_policy_badge(self, rank: int, probability: float, target: tuple[int, int], color: Color) -> None:
        square_size = Dimension.SQUARE_SIZE.value
        target_row, target_col = target
        center = (target_col * square_size + square_size - 12, target_row * square_size + 12)
        if rank == 1:
            label = f"{probability:.0%}"
            text_surface = self._debug_font.render(label, True, ARMY_LIGHT)
            badge_width = max(34, text_surface.get_width() + 10)
            badge_rect = pygame.Rect(0, 0, badge_width, 18)
            badge_rect.center = center
            badge_rect.x = max(2, min(badge_rect.x, self.display_grid_width - badge_rect.width - 2))
            badge_rect.y = max(2, min(badge_rect.y, self.display_grid_height - badge_rect.height - 2))
            pygame.draw.rect(self.game_area, color, badge_rect, border_radius=9)
            pygame.draw.rect(self.game_area, ARMY_LIGHT, badge_rect, 1, border_radius=9)
            self.game_area.blit(text_surface, text_surface.get_rect(center=badge_rect.center))
            return

        pygame.draw.circle(self.game_area, color, center, 11)
        pygame.draw.circle(self.game_area, ARMY_LIGHT, center, 11, 1)
        text_surface = self._preview_font.render(str(rank), True, ARMY_LIGHT)
        self.game_area.blit(text_surface, text_surface.get_rect(center=center))

    def _blend_color(self, first: Color, second: Color, weight: float) -> Color:
        return tuple(int(first[i] * (1.0 - weight) + second[i] * weight) for i in range(3))

    def draw_tile_types(self):
        """
        Draw tile type labels in the upper-right corner of each tile.
        Types: 0=empty, -2=mountain, 1=general0, 2=general1, 40-50=city
        """
        square_size = Dimension.SQUARE_SIZE.value
        channels = self.game.channels
        agents = self.game.agents

        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Determine tile type
                if channels.mountains[i, j]:
                    tile_type = "-2"
                elif channels.generals[i, j]:
                    if channels.ownership[agents[0]][i, j]:
                        tile_type = "1"
                    else:
                        tile_type = "2"
                elif channels.cities[i, j]:
                    tile_type = "C"  # City
                else:
                    tile_type = "0"

                # Render the type label in upper-right corner
                text_surface = self._debug_font.render(tile_type, True, (0, 255, 0))  # Green text
                text_rect = text_surface.get_rect()
                # Position in upper-right with small padding
                x_pos = square_size - text_rect.width - 2
                y_pos = 2
                self.tiles[i][j].blit(text_surface, (x_pos, y_pos))
