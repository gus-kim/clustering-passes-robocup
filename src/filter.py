import csv
from dataclasses import dataclass
import math
from typing import List, Tuple, Optional, Dict
import numpy as np

@dataclass
class PlayerData:
    x: float
    y: float
    vx: float
    vy: float
    body: float
    neck: float

@dataclass
class GameState:
    cycle: int
    stopped: bool
    playmode: str
    left_team_name: str
    left_score: int
    left_pen_score: int
    right_team_name: str
    right_score: int
    right_pen_score: int
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    left_team: List[PlayerData]
    right_team: List[PlayerData]

class SoccerFeatureExtractor:
    def __init__(self):
        self.game_states = []
        self.features = []
        self.VIEW_DISTANCE = 50.0
        self.POSSESSION_DISTANCE = 1.0
        self.PASS_CONSIDER_DISTANCE = 50.0
    
    def load_csv(self, file_path: str):
        with open(file_path, 'r') as csvfile:
            # Pular linhas de comentário
            while True:
                pos = csvfile.tell()
                line = csvfile.readline()
                if not line.startswith('#'):
                    csvfile.seek(pos)
                    break
            
            reader = csv.reader(csvfile)
            header = next(reader)  # Ler cabeçalho
            
            for row in reader:
                self._process_row(row)
    
    def _process_row(self, row):
        # Extrair dados gerais do jogo
        cycle = int(row[1])
        stopped = bool(int(row[2]))
        playmode = row[3]
        left_team_name = row[4]
        left_score = int(row[5])
        left_pen_score = int(row[6])
        right_team_name = row[7]
        right_score = int(row[8])
        right_pen_score = int(row[9])
        ball_x = float(row[10])
        ball_y = float(row[11])
        ball_vx = float(row[12])
        ball_vy = float(row[13])
        
        # Processar time da esquerda (11 jogadores)
        left_team = []
        for i in range(11):
            start_idx = 14 + i*7  # Cada jogador tem 7 campos (t, x, y, vx, vy, body, neck)
            player_data = PlayerData(
                x=float(row[start_idx+1]),
                y=float(row[start_idx+2]),
                vx=float(row[start_idx+3]),
                vy=float(row[start_idx+4]),
                body=float(row[start_idx+5]),
                neck=float(row[start_idx+6])
            )
            left_team.append(player_data)
        
        # Processar time da direita (11 jogadores)
        right_team = []
        for i in range(11):
            start_idx = 91 + i*7  # 13 + 11*7 = 90
            player_data = PlayerData(
                x=float(row[start_idx+1]),
                y=float(row[start_idx+2]),
                vx=float(row[start_idx+3]),
                vy=float(row[start_idx+4]),
                body=float(row[start_idx+5]),
                neck=float(row[start_idx+6])
            )
            right_team.append(player_data)
        
        # Criar e armazenar o estado do jogo
        game_state = GameState(
            cycle=cycle,
            stopped=stopped,
            playmode=playmode,
            left_team_name=left_team_name,
            left_score=left_score,
            left_pen_score=left_pen_score,
            right_team_name=right_team_name,
            right_score=right_score,
            right_pen_score=right_pen_score,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_vx=ball_vx,
            ball_vy=ball_vy,
            left_team=left_team,
            right_team=right_team
        )
        
        self.game_states.append(game_state)
    
    def extract_features(self, team: str = 'left'):
        """Extrai features relacionadas a passes para cada ciclo"""
        for i in range(len(self.game_states)):
            current_state = self.game_states[i]
            prev_state = self.game_states[i-1] if i > 0 else None
            
            features = {
                'cycle': current_state.cycle,
                'team': team,
                'playmode': current_state.playmode,
                'ball_x': current_state.ball_x,
                'ball_y': current_state.ball_y,
                'ball_speed': math.hypot(current_state.ball_vx, current_state.ball_vy),
                'ball_acceleration': 0.0,
                'possession_change': 0,
                'player_with_ball': -1,
                'teammates_in_pass_range': 0,
                'opponents_in_pass_range': 0,
                'best_pass_target': -1,
                'best_pass_alignment': 0.0,
                'best_pass_distance': 0.0,
                'pressure_on_ball': 0,
                'space_around_ball': 0.0,
                'field_zone': self._get_field_zone(current_state.ball_x, current_state.ball_y),
                'is_pass': 0  # Será preenchido posteriormente
            }
            
            # Preencher features que dependem do estado anterior
            if prev_state is not None:
                # Aceleração da bola
                prev_speed = math.hypot(prev_state.ball_vx, prev_state.ball_vy)
                curr_speed = features['ball_speed']
                features['ball_acceleration'] = curr_speed - prev_speed
                
                # Mudança de posse
                prev_possessor = self._get_ball_possessor(prev_state, team)
                curr_possessor = self._get_ball_possessor(current_state, team)
                
                if prev_possessor != curr_possessor:
                    features['possession_change'] = 1
            
            # Jogador com a bola
            possessor = self._get_ball_possessor(current_state, team)
            if possessor is not None:
                features['player_with_ball'] = possessor
                
                # Características do jogador com a bola
                player = current_state.left_team[possessor-1] if team == 'left' else current_state.right_team[possessor-1]
                features.update({
                    'player_speed': math.hypot(player.vx, player.vy),
                    'player_body_angle': player.body,
                    'player_neck_angle': player.neck
                })
                
                # Melhor alvo de passe
                best_target, best_alignment, best_distance = self._find_best_pass_target(
                    current_state, team, possessor
                )
                
                if best_target is not None:
                    features.update({
                        'best_pass_target': best_target,
                        'best_pass_alignment': best_alignment,
                        'best_pass_distance': best_distance,
                        'teammates_in_pass_range': self._count_teammates_in_pass_range(
                            current_state, team, possessor
                        )
                    })
            
            # Pressão sobre a bola
            features['pressure_on_ball'] = self._calculate_pressure_on_ball(current_state, team)
            
            # Espaço ao redor da bola
            features['space_around_ball'] = self._calculate_space_around_ball(current_state)
            
            # Adicionar ao conjunto de features
            self.features.append(features)
        
        # Pós-processamento para marcar passes (usando critérios similares ao detector original)
        self._mark_passes(team)
    
    def _get_ball_possessor(self, state: GameState, team: str) -> Optional[int]:
        """Retorna o número do jogador que tem posse da bola ou None"""
        team_players = state.left_team if team == 'left' else state.right_team
        ball_pos = (state.ball_x, state.ball_y)
        
        for i, player in enumerate(team_players, start=1):
            player_pos = (player.x, player.y)
            if self._distance(player_pos, ball_pos) < self.POSSESSION_DISTANCE:
                return i
        return None
    
    def _find_best_pass_target(self, state: GameState, team: str, passer_num: int) -> Tuple[Optional[int], float, float]:
        """Encontra o melhor alvo de passe para o jogador com a bola"""
        team_players = state.left_team if team == 'left' else state.right_team
        passer_pos = (team_players[passer_num-1].x, team_players[passer_num-1].y)
        
        best_target = None
        best_alignment = 0.0
        best_distance = 0.0
        
        for i, player in enumerate(team_players, start=1):
            if i == passer_num:
                continue
                
            player_pos = (player.x, player.y)
            dist = self._distance(passer_pos, player_pos)
            
            if dist > self.PASS_CONSIDER_DISTANCE:
                continue
                
            # Vetor do passador para o possível receptor
            to_player_vec = (player_pos[0] - passer_pos[0], player_pos[1] - passer_pos[1])
            norm_to_player = (to_player_vec[0]/dist, to_player_vec[1]/dist)
            
            # Direção do corpo do passador (convertido para vetor unitário)
            body_angle = math.radians(team_players[passer_num-1].body)
            body_vec = (math.cos(body_angle), math.sin(body_angle))
            
            # Alinhamento com a direção do corpo
            alignment = body_vec[0] * norm_to_player[0] + body_vec[1] * norm_to_player[1]
            
            if alignment > best_alignment:
                best_target = i
                best_alignment = alignment
                best_distance = dist
        
        return best_target, best_alignment, best_distance
    
    def _count_teammates_in_pass_range(self, state: GameState, team: str, passer_num: int) -> int:
        """Conta quantos companheiros estão em posição de receber um passe"""
        team_players = state.left_team if team == 'left' else state.right_team
        passer_pos = (team_players[passer_num-1].x, team_players[passer_num-1].y)
        count = 0
        
        for i, player in enumerate(team_players, start=1):
            if i == passer_num:
                continue
                
            player_pos = (player.x, player.y)
            dist = self._distance(passer_pos, player_pos)
            
            if dist < self.PASS_CONSIDER_DISTANCE:
                count += 1
                
        return count
    
    def _calculate_pressure_on_ball(self, state: GameState, team: str) -> int:
        """Calcula quantos oponentes estão próximos da bola"""
        opponent_team = state.right_team if team == 'left' else state.left_team
        ball_pos = (state.ball_x, state.ball_y)
        count = 0
        
        for player in opponent_team:
            player_pos = (player.x, player.y)
            if self._distance(player_pos, ball_pos) < 5.0:  # Raio de pressão
                count += 1
                
        return count
    
    def _calculate_space_around_ball(self, state: GameState) -> float:
        """Calcula o espaço médio ao redor da bola"""
        ball_pos = (state.ball_x, state.ball_y)
        distances = []
        
        # Considera todos os jogadores
        for player in state.left_team + state.right_team:
            player_pos = (player.x, player.y)
            distances.append(self._distance(player_pos, ball_pos))
        
        if not distances:
            return 0.0
            
        return np.mean(distances)
    
    def _get_field_zone(self, x: float, y: float) -> int:
        """Divide o campo em zonas (1-6)"""
        if x < -30:
            return 1  # Defesa
        elif x < 0:
            return 2  # Meio-campo defensivo
        elif x < 30:
            return 3  # Meio-campo ofensivo
        else:
            return 4  # Ataque
        
        # Poderia ser mais granular incluindo também a posição y
    
    def _mark_passes(self, team: str):
        """Marca os ciclos onde ocorreram passes (para criar labels)"""
        for i in range(1, len(self.features)):
            curr = self.features[i]
            prev = self.features[i-1]
            
            # Critérios para marcar como passe:
            # 1. Mudança brusca na velocidade da bola
            # 2. Jogador do time tinha a bola no ciclo anterior
            # 3. Aceleração positiva significativa
            if (prev['player_with_ball'] != -1 and
                curr['ball_acceleration'] > 0.5 and
                curr['ball_speed'] > 1.0 and
                curr['possession_change'] == 1):
                
                self.features[i]['is_pass'] = 1
    
    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calcula a distância euclidiana entre dois pontos"""
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
    
    def save_features_to_csv(self, output_file: str):
        """Salva as features extraídas em um novo arquivo CSV"""
        if not self.features:
            raise ValueError("Nenhuma feature foi extraída ainda. Chame extract_features() primeiro.")
        
        fieldnames = list(self.features[0].keys())
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for feature_set in self.features:
                writer.writerow(feature_set)

import argparse

def main():
    """
    Função principal para executar o script a partir da linha de comando.
    """
    parser = argparse.ArgumentParser(description="Extrai features de passes de arquivos CSV de log de futebol de robôs.")
    parser.add_argument("input_file", help="Caminho para o arquivo CSV de entrada (da pasta CSV_Completo).")
    parser.add_argument("output_file", help="Caminho para o arquivo CSV de saída (onde salvar as features).")
    parser.add_argument("--team", type=str, default="left", choices=["left", "right"],
                        help="Time para o qual extrair as features ('left' ou 'right').")
    
    args = parser.parse_args()
    
    print(f"Processando arquivo: {args.input_file}")
    print(f"Time selecionado: {args.team}")

    # Processar o arquivo de log
    processor = SoccerFeatureExtractor()
    processor.load_csv(args.input_file)
    
    # Extrair features para o time especificado
    processor.extract_features(team=args.team)
    
    # Salvar as features em um novo arquivo CSV
    processor.save_features_to_csv(args.output_file)
    
    print(f"Features extraídas e salvas com sucesso em '{args.output_file}'")

if __name__ == "__main__":
    main()