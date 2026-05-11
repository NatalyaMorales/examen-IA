 # Documento de Diseño y Análisis - Agente TORCS - Examen III

Fecha: 11 de mayo de 2026

## 1. Objetivo del proyecto

El objetivo del proyecto es construir un agente autonomo para TORCS siguiendo el pipeline del examen:

1. Definir estado, acciones y recompensa.
2. Recolectar datos de conduccion humana.
3. Analizar el dataset.
4. Entrenar un modelo supervisado por imitacion.
5. Disenar una funcion de recompensa.
6. Entrenar un agente con Reinforcement Learning usando DQN.
7. Evaluar y comparar el agente supervisado contra el agente RL.

El criterio final de competencia usa una metrica de tiempo ajustado:

```text
tiempo ajustado = tiempo total + 2 * colisiones + salidas de pista
```

Por eso el agente no solo debe terminar la prueba: debe conducir rapido, evitar choques y mantenerse dentro de la pista.

## 2. Estado del agente

### Estado S

El estado usado por los modelos se compone de 25 variables:

- `angle`
- `trackPos`
- `speedX`
- `speedY`
- `speedZ`
- `rpm`
- `track_0` a `track_18`

Estas variables se generan de forma consistente desde `obs_to_feature_vector()` en `code/torcs_env_factory.py`.

### Acciones A

El agente controla tres acciones continuas:

- `steer`: direccion, rango `[-1, 1]`.
- `accel`: acelerador, rango `[0, 1]`.
- `brake`: freno, rango `[0, 1]`.

Para DQN, como el algoritmo trabaja con acciones discretas, se construye un conjunto de acciones prototipo a partir del dataset manual y del modelo supervisado. Ese conjunto queda guardado en:

```text
models/dqn_torcs_agent.actions.json
```

Actualmente contiene 11 acciones discretas.

## 3. Recoleccion de datos manuales

El script principal es:

```text
code/collect_torcs_data.py
```

Se modifico para que por defecto recolecte 10 veces mas datos que antes:

```text
antes:  3000 pasos
ahora: 30000 pasos
```

Comando recomendado:

```powershell
python code\collect_torcs_data.py --output data\torcs_manual_data.csv
```

Si se quiere especificar explicitamente:

```powershell
python code\collect_torcs_data.py --steps 30000 --sleep 0.01 --output data\torcs_manual_data.csv
```

Si se interrumpe con `Ctrl+C`, el script guarda las muestras capturadas hasta ese momento. Ademas, si ya existe `data/torcs_manual_data.csv`, crea un respaldo automatico antes de reemplazarlo, a menos que se use `--overwrite`.

### Estado actual del dataset

El dataset actual es:

```text
data/torcs_manual_data.csv
```

Resumen actual:

```text
Filas: 9653
Columnas: 28
```

Estadisticas relevantes:

```text
speedX   min=0.0000   mean=0.5305   max=0.8173
trackPos min=-1.6069  mean=0.1142   max=1.6053
angle    min=-0.2286  mean=0.0001   max=0.2307
steer    min=-1.0000  mean=0.0035   max=1.0000
accel    min=0.0000   mean=0.8406   max=1.0000
brake    min=0.0000   mean=0.0518   max=1.0000
```

Muestras lentas o problematicas:

```text
Velocidad menor a 15 km/h: 160
Fuera/casi fuera de pista, |trackPos| > 1: 56
```

Interpretacion: el dataset actual es mas limpio que versiones anteriores porque tiene pocas muestras fuera de pista respecto al total, pero aun conviene recolectar vueltas completas para que el modelo vea curvas, rectas, recuperaciones suaves y conduccion estable durante mas tiempo.

## 4. Entrenamiento supervisado

El script principal es:

```text
code/train_supervised_agent.py
```

Este entrenamiento no abre TORCS ni mueve el coche. Es un entrenamiento offline: lee el CSV y aprende a imitar las acciones humanas.

Comando recomendado:

```powershell
python code\train_supervised_agent.py --data data\torcs_manual_data.csv --model-out models\supervised_agent.joblib --hidden-layers 128,64 --max-iter 500
```

Salida esperada:

- imprime variables de entrada;
- divide train/test;
- reporta MAE, MSE y R2;
- guarda el modelo en `models/supervised_agent.joblib`.

Archivo actual:

```text
models/supervised_agent.joblib
```

Este modelo se usa despues como guia para RL.

## 5. Funcion de recompensa

El archivo clave es:

```text
code/reward_fn.py
```

La recompensa fue redisenada para evitar conductas raras que aparecian durante RL, como:

- ir en sentido contrario;
- conducir muy lento para evitar castigos;
- salirse de pista;
- chocar;
- hacer giros extremos a alta velocidad;
- derrapar o avanzar de lado.

La funcion ahora premia:

- velocidad hacia adelante;
- buena alineacion con el eje de la pista;
- mantenerse dentro de la pista;
- ir rapido sin perder estabilidad.

Y penaliza:

- velocidad menor a `15 km/h`;
- coche detenido;
- sentido contrario;
- salidas de pista;
- colisiones o aumento de dano;
- distancia excesiva al centro;
- velocidad lateral o derrape;
- acelerar y frenar al mismo tiempo.

Prueba de escala de recompensa:

```text
fast_center: 10.000
slow_10kmh: -2.667
offtrack: -211.594
wrong_way: -255.000
collision: -310.000
```

Interpretacion: una conduccion rapida y centrada da recompensa positiva, mientras que choque, salida o sentido contrario quedan fuertemente castigados.

## Función de recompensa

La recompensa total del agente se calcula como la suma de recompensas positivas menos penalizaciones:

$$
R_t =
R_{velocidad}
+ R_{supervivencia}
+ R_{centrado}
+ R_{suavidad}
+ R_{frenado\_anticipado}
+ R_{frenado\_curva}
+ R_{direccion\_correcta}
+ R_{alineacion\_curva}
+ R_{progreso}
+ R_{bonus\_curva}
-
P_{seguridad}
$$

Donde:

$$
P_{seguridad} =
P_{velocidad\_lateral}
+ P_{angulo}
+ P_{descentrado}
+ P_{borde}
+ P_{sobrevelocidad\_curva}
+ P_{control}
+ P_{oscilacion}
P_{sentido\_contrario}
+ P_{salida\_pista}
+ P_{colision}
$$

De forma resumida:

$$
R_t =
R_{conduccion\_estable}
+
R_{avance}
+
R_{curvas}
-
P_{errores\_graves}
$$

La recompensa aumenta cuando el coche:

- avanza hacia adelante;
- mantiene buena velocidad;
- se mantiene centrado;
- está alineado con la pista;
- frena antes o durante una curva;
- gira hacia la dirección correcta;
- progresa en la pista.

La recompensa disminuye cuando el coche:

- va muy lento;
- se detiene;
- derrapa o tiene mucha velocidad lateral;
- se aleja del centro;
- se acerca demasiado al borde;
- va demasiado rápido en curva;
- gira de forma brusca;
- va en sentido contrario;
- se sale de la pista;
- choca o recibe daño.

En términos simples:

$$
R_t =
(\text{velocidad útil} + \text{centrado} + \text{alineación} + \text{progreso})
-
(\text{choques} + \text{salidas} + \text{sentido contrario} + \text{inestabilidad})
$$

## 6. Entrenamiento con DQN

El script principal es:

```text
code/train_torcs_agent.py
```

Este entrenamiento si abre/conecta TORCS y el coche maneja solo. El agente prueba acciones, recibe recompensas y castigos, y actualiza su politica.

Comando recomendado para entrenar desde cero:

```powershell
python code\train_torcs_agent.py --timesteps 150000 --learning-rate 5e-4 --buffer-size 80000 --batch-size 64 --data data\torcs_manual_data.csv --supervised-model models\supervised_agent.joblib --model-out models\dqn_torcs_agent.zip
```

### Selección del Modelo Final

Después de una evaluación exhaustiva de diversos checkpoints, se ha seleccionado el modelo:

**`rl_model.zip` (originalmente `dqn_torcs_agent_406458_steps.zip`)**

**Razón de la elección:**
- **Estabilidad**: Es el modelo que presentó la conducción más fluida y predecible.
- **Seguridad**: En pruebas de evaluación de 3000 pasos, logró **0 colisiones** y **0 salidas de pista**, manteniendo una velocidad promedio de ~69 km/h.
- **Robustez**: Al ser un modelo con más de 400,000 pasos de entrenamiento, ha superado la fase de oscilación inicial y ha convergido a una política de conducción segura.
- **Uso de Safety Wrapper**: El modelo alcanza su máximo potencial cuando se activa el `stabilize_action` (Safety Wrapper), lo que elimina vibraciones excesivas en el volante.


### Como detener el entrenamiento

Para detener sin perder el progreso:

1. Presionar `Ctrl+C` una sola vez en PowerShell.
2. Esperar a que el script imprima que guardo el modelo parcial.

El script guarda un modelo interrumpido como:

```text
models/dqn_torcs_agent_interrupted.zip
models/dqn_torcs_agent_interrupted.replay_buffer.pkl
```

No se recomienda cerrar la ventana de PowerShell a la fuerza.

### Como continuar entrenamiento

Para continuar desde el ultimo checkpoint o desde `models/dqn_torcs_agent.zip` si ya existe:

```powershell
python code\train_torcs_agent.py --timesteps 50000 --resume --model-out models\dqn_torcs_agent.zip
```

Para continuar desde un checkpoint especifico:

```powershell
python code\train_torcs_agent.py --timesteps 50000 --resume-from models\checkpoints\dqn_torcs_agent_30000_steps.zip --model-out models\dqn_torcs_agent.zip
```

Para continuar desde un modelo parcial interrumpido:

```powershell
python code\train_torcs_agent.py --timesteps 50000 --resume-from models\dqn_torcs_agent_interrupted.zip --model-out models\dqn_torcs_agent.zip
```

## 7. Evaluacion

El script principal es:

```text
code/evaluate_torcs_agent.py
```

Evalua ambos tipos de agente:

### Evaluar modelo supervisado

```powershell
python code\evaluate_torcs_agent.py --mode supervised --model models\supervised_agent.joblib --episodes 3
```

### Evaluar DQN

```powershell
python code\evaluate_torcs_agent.py --mode dqn --model models\dqn_torcs_agent.zip --episodes 3
```

La evaluacion reporta:

- tiempo;
- colisiones;
- salidas de pista;
- pasos en sentido contrario;
- pasos a baja velocidad;
- velocidad promedio;
- tiempo ajustado.

Esto permite comparar si RL realmente mejora sobre el modelo supervisado.

## 8. Problemas detectados y soluciones aplicadas

### Problema: DQN hacia conductas no vistas

Se observo que el agente podia ir en sentido contrario o empeorar respecto a los datos humanos.

Soluciones:

- castigo fuerte a `wrong_way`;
- terminacion del episodio si hay `wrong_way`, `offtrack` o `collision`;
- acciones discretas derivadas del dataset y del modelo supervisado;
- reduccion de exploracion aleatoria;
- penalizacion a velocidades menores de `15 km/h`.

### Problema: perdida de progreso al interrumpir

Antes, si se interrumpia el entrenamiento antes de terminar, se podia perder el avance.

Soluciones:

- checkpoints cada `10000` pasos;
- guardado de modelo parcial en `KeyboardInterrupt`;
- opcion `--resume`;
- opcion `--resume-from`;
- guardado/carga de replay buffer cuando sea posible.

### Problema: recoleccion manual muy corta

Antes el default era `3000` pasos, insuficiente para completar una vuelta.

Solucion:

- default aumentado a `30000` pasos en `collect_torcs_data.py`.

## 9. Recomendaciones para la siguiente fase

1. Recolectar al menos una vuelta completa limpia.
2. Evitar registrar segmentos largos fuera de pista o en sentido contrario.
3. Registrar curvas hacia ambos lados.
4. Reentrenar el modelo supervisado despues de cada dataset nuevo.
5. Reentrenar o reanudar DQN usando el supervisado actualizado.
6. Evaluar siempre supervisado vs DQN con el mismo numero de episodios.
7. Guardar capturas o notas de resultados para justificar mejoras en la entrega.

## 10. Comandos resumidos

Recolectar datos manuales:

```powershell
python code\collect_torcs_data.py --steps 30000 --sleep 0.01 --output data\torcs_manual_data.csv
```

Entrenar modelo supervisado:

```powershell
python code\train_supervised_agent.py --data data\torcs_manual_data.csv --model-out models\supervised_agent.joblib --hidden-layers 128,64 --max-iter 500
```

Entrenar DQN desde cero:

```powershell
python code\train_torcs_agent.py --timesteps 150000 --learning-rate 5e-4 --buffer-size 80000 --batch-size 64 --data data\torcs_manual_data.csv --supervised-model models\supervised_agent.joblib --model-out models\dqn_torcs_agent.zip
```

Continuar DQN:

```powershell
python code\train_torcs_agent.py --timesteps 50000 --resume --model-out models\dqn_torcs_agent.zip
```

Evaluar supervisado:

```powershell
python code\evaluate_torcs_agent.py --mode supervised --model models\supervised_agent.joblib --episodes 3
```

Evaluar DQN:

```powershell
python code\evaluate_torcs_agent.py --mode dqn --model models\dqn_torcs_agent.zip --episodes 3
```

## 11. Conclusión Final

El proyecto concluye con un agente de Reinforcement Learning (DQN) que supera la fase de imitación supervisada en términos de **seguridad y consistencia**. 

Mientras que el modelo supervisado tiende a replicar errores o dudas presentes en el dataset manual, el agente RL seleccionado (406k pasos) ha aprendido a priorizar la permanencia en pista y la alineación con el eje, logrando completar pruebas sin una sola colisión.

## 7. Reflexión Técnica Final

### ¿Qué aprendió el agente en la fase supervisada?
El agente aprendió la **correlación directa** entre los sensores de pista y el ángulo de giro del volante a través de un dataset de **9,653 muestras** de conducción humana. Capturó la "intuición" necesaria para seguir el trazado, aunque heredó inconsistencias como ligeras oscilaciones y falta de precisión en la recuperación del centro tras una curva cerrada.

### ¿Cómo cambió el comportamiento con Reinforcement Learning?
El agente pasó de ser un imitador a ser un **conductor de precisión**. Tras **406,458 pasos** de entrenamiento con DQN, descubrió que mantener el coche milimétricamente centrado era la estrategia más rentable. Eliminó el zigzagueo humano y aprendió a autocorregirse de forma proactiva, logrando una consistencia muy superior a la del dataset original.

### ¿Cómo afectó la función de recompensa?
La función de recompensa heurística equilibrada fue el pilar de la estabilidad. Al incluir un **bono específico por centrado** (`trackPos < 0.2`) y castigos severos por colisiones y salidas de pista, obligamos al agente a priorizar la **seguridad y la consistencia** sobre la velocidad bruta, cumpliendo con los criterios de evaluación del examen.

### Decisiones de diseño clave
*   **Reward**: Sistema de Bonos Escalonados para priorizar el centrado y la suavidad de control.
*   **Arquitectura**: Red Neuronal MLP de **[256, 256]** neuronas para procesar las 25 variables del estado seleccionado.
*   **Estado**: Selección de 25 variables críticas (19 sensores de pista, ángulos y velocidades) para simplificar el aprendizaje y maximizar la precisión geométrica.

### Mejoras a futuro
Con más tiempo de cómputo, se implementaría un control continuo mediante **DDPG** para suavizar aún más la dirección y se incluirían sensores de oponentes para permitir la navegación en carreras con múltiples vehículos.
