# EmbracenetFuzzy

Reconocimiento de emociones multimodal (audio, video, texto) con **EmbraceNet** +
capa difusa: experimento temprano de fusión de la tesis, anterior al repo
[fuzzy-embracenet](https://github.com/darwinrocha85/fuzzy-embracenet) (activación
difusa gaussiana + fusión adaptativa).

## Estructura
| Carpeta/archivo | Qué es |
|---|---|
| `EmbracenetFuzzy.py` | Entrada principal |
| `Models/` | `Embracenet.py`, `WeightedSum.py` |
| `Datasets/` | Cargadores IEMOCAP y AFFWILD2 |
| `Utils/` | Entrenamiento, dataloaders, guardado de resultados |
| `Data/` | Features preextraídos por modalidad (`.pkl`, `.csv`) |
| `Results/` | CSVs de entrenamiento |

## Dataset
IEMOCAP y AFFWILD2 (solicitar acceso en sus webs; no incluidos por licencia).

> Sin artículo publicado: manuscrito en evaluación de revista, sin link de paper.
