# RESULTS — Altur HackMTY 2026

## Estado de las puertas

Fase 1 aprobada. Fase 2 tiene evaluación HTTP local reproducida y pruebas de bordes, pero **la puerta completa sigue cerrada**: falta verificar el endpoint público. Docker fue aplazado por el usuario y no bloquea esta fase. No se han iniciado las fases 3–6.

| Variante/fase | Accuracy | AUC | EER | Brier | Estado |
| --- | ---: | ---: | ---: | ---: | --- |
| Referencia: turnos oficiales | 0.957746 (68/71) | 0.980922 | 0.054054 | 0.035343 | Reproducida; no desplegable sin VAD |
| VAD original sin ajustar | 0.915493 (65/71) | 0.959459 | 0.117647 | 0.080870 | No alcanza la puerta |
| Fase 1: VAD seleccionado en train | 0.929577 (66/71) | 0.984897 | 0.054054 | 0.045284 | APROBADA |
| Fase 2: HTTP local | 0.929577 (66/71) | 0.984897 | 0.054054 | 0.045284 | Equivalencia comprobada |
| Fase 2: entorno Python limpio | 0.929577 (66/71) | 0.984897 | 0.054054 | 0.045284 | Equivalencia comprobada |
| Fase 2: host público | — | — | — | — | Pendiente de despliegue y verificación |
| Docker | — | — | — | — | Aplazado por el usuario; fuera de la puerta actual |
| Fase 3: comportamiento adicional | — | — | — | — | No iniciada: puerta anterior pendiente |
| Fase 4: semántica | — | — | — | — | No iniciada |
| Fase 5: robustez/calibración | — | — | — | — | No iniciada |
| Fase 6: demo | — | — | — | — | No iniciada |

Todas las métricas medidas corresponden a las mismas 71 llamadas de val, separadas por hablante de train. Positivo = sintético. EER se interpola en el cruce FAR/FRR; los JSON también incluyen la definición discreta del código original. Las cifras calibradas aportadas en el brief no se presentan como reproducidas: la calibración no se ha ejecutado.

## Datos y baseline preservado

- Repo oficial leído antes de ejecutar: `alturio/hackmty26`, commit `26b519598a1520cf6306d78902ef5047ae670aa4`, release `v1.0`.
- 353 llamadas: train 282 (113 humanas, 169 sintéticas); val 71 (37 humanas, 34 sintéticas). Cada una tiene WAV y JSON; los 353 WAV cumplen estéreo, 8 kHz y PCM16.
- `features.py` contiene **58 predictores**, más tres columnas de identificación/etiqueta/split en su CSV: 61 columnas en total.
- Se conservaron las definiciones originales. La refactorización para recibir turnos en memoria y el VAD con configuración por defecto dieron resultados idénticos al código adjunto para las 353 llamadas.
- El código original de entrenamiento terminaba ajustando con train+val. Se corrigió: el modelo entregado usa exclusivamente train. No se calculó CV aleatoria ni se atribuye separación por hablante a folds sin identificadores de hablante.

## Fase 1: VAD

Se probaron **540 configuraciones**, todas evaluadas por correlación contra referencias de las 282 llamadas de train. Val no intervino en la selección. Se congeló la configuración antes de generar las predicciones del modelo seleccionado en val.

Malla: `frame_ms ∈ {20,30}`, `thresh_db ∈ {-46,-42,-38,-34,-30,-26}`, `min_speech ∈ {0.1,0.2,0.3}`, `min_sil ∈ {0.1,0.2,0.25,0.3,0.4}`, `noise_margin ∈ {6,12,18}`.

Configuración elegida:

```json
{
  "frame_ms": 30,
  "thresh_db": -46.0,
  "min_speech": 0.1,
  "min_sil": 0.2,
  "noise_margin": 18.0
}
```

El umbral de actividad es el máximo entre `thresh_db` y el percentil 10 de energía del canal más `noise_margin`. Se maximiza Pearson de `lat_med`, con correlación media de las cinco variables prioritarias como desempate.

| Variable | Pearson train (n=282) | Pearson val (n=71) | MAE train |
| --- | ---: | ---: | ---: |
| `lat_med` | 0.947717 | 0.947633 | 0.348936 |
| `lat_mean` | 0.922605 | 0.925372 | 0.516333 |
| `barge_in_rate` | 0.894513 | 0.875019 | 0.059403 |
| `overlap_ratio` | 0.887432 | 0.842966 | 0.030436 |
| `n_caller` | 0.792691 | 0.752486 | 3.918440 |

Las correlaciones de las 58 variables están en `reports/vad_correlations_train.csv` y `reports/vad_correlations_val.csv`. La malla completa está en `reports/vad_grid_train.csv`.

Puerta: `lat_med` r = 0.947717 > 0.9; accuracy = 0.929577 ≥ 0.928. Son dos aciertos menos que el baseline de referencia (2.82 puntos porcentuales), dentro del margen permitido. No fue necesario probar WebRTC/Silero para satisfacer esta puerta.

Bleed: en tramos de train donde la referencia marca solo agente, **0 llamadas** superaron simultáneamente |correlación instantánea| > 0.6 y |coeficiente lineal| > 0.003. No se aplicó resta entre canales. Esta prueba no descarta filtración con retardo o no lineal. Los turnos de referencia se usan solo para este diagnóstico y la comparación, nunca para inferencia.

## Fase 2: endpoint

`POST /detect` recibe el WAV base64, extrae los turnos con el VAD congelado, obtiene las mismas 58 variables y devuelve el veredicto y la confianza de esa clase. No usa JSON de referencia ni imputa NaN de forma distinta al entrenamiento.

- 71/71 peticiones completas; 0 abstenciones en val.
- Diferencia máxima entre probabilidades HTTP y offline: `1.3270634591222574e-16`.
- Pruebas de bordes: 17 aprobadas. Incluyen mono, audio corto, silencio, cero frames, menos de cuatro turnos, un solo canal con actividad, WAV truncado, base64/JSON inválidos y formato incompatible.
- `confidence=0.5` significa abstención; el bool `false` solo satisface el esquema obligatorio.
- Confianza sin redondeo; AUC/Brier se calculan con `P(synthetic)` recuperada del veredicto y su confianza.
- El endpoint procesa audio en memoria, no guarda muestras y evita reflejar base64 en errores. Límite: 32 MiB y 900 s.

| Ejecución | p50 | p95 |
| --- | ---: | ---: |
| HTTP local inicial | 54.12 ms | 103.54 ms |
| HTTP con servidor en entorno limpio | 51.21 ms | 86.00 ms |

Medición secuencial con servidor caliente y HTTP loopback: incluye serialización JSON, transporte e inferencia; excluye lectura del archivo y codificación base64. Estas cifras **no son latencias de internet ni prueban concurrencia o disponibilidad de 15 minutos**. El entorno limpio se creó sin paquetes heredados y solo con las dependencias del servidor. Docker se aplaza por indicación del usuario.

Modelo evaluado SHA-256: `135723cf56e554b006b224430d2a5dcb2d6f2950d7e86f75d28f70615ff8ea69`.

Código del equipo: [Chaarliee06/altur-detector](https://github.com/Chaarliee06/altur-detector). Render está configurado para Python 3.12 con dependencias fijadas y el modelo incluido; no descarga el dataset.
Host público: pendiente de despliegue y verificación. `/health` identifica tanto el modelo como el código de inferencia. La prueba pública exige equivalencia HTTP local de ese mismo código y 900 segundos de respuestas estables.

Para cumplir la restricción de procesamiento local, las 71 llamadas solo se envían por HTTP loopback. Las pruebas de un host externo usan tonos generados y comprueban el mismo SHA-256. No se ha enviado ningún audio del dataset a un host externo.

## Límites actuales del argumento

La validación respeta el split oficial, pero no demuestra rendimiento sobre voces del set oculto. Todavía no se han medido motores con menor latencia, clips parciales, ruido/ganancia, calibración adicional ni señales semánticas. No usar las cifras de duración del baseline de referencia como resultados del endpoint con VAD.

`make all` reproduce las etapas implementadas y se detiene al encontrar una puerta pendiente. El paquete contiene código, modelo y resultados agregados; excluye audio, turnos oficiales, manifiesto y cachés por llamada.
