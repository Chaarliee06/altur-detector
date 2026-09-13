# RESULTS — Altur HackMTY 2026

## Corrección urgente del contrato del juez — 2026-09-13

El cliente oficial actual envía `audio_base64`, `call_id`, `sample_rate` y `channels`. Se añadió compatibilidad con ese contrato y con `audio`; metadatos opcionales ignorados, respuestas de respaldo HTTP 200, booleano nativo y CORS en respuestas normales y de error. Se probaron errores de JSON/base64/WAV, campos ausentes, excepciones de inferencia, límite del cuerpo y preflight. Son comprobaciones de desarrollo; la puerta válida sigue siendo el cliente oficial contra Render.

**Despliegue y verificación pública del contrato nuevo: APROBADOS.** Render sirve el commit `19290711b7f31a9270ca88303c42c14515742e28`, despliegue `dep-daj2f6u7bikc73agm5l0`, confirmado `live` el 2026-09-13. `/health` respondió HTTP 200 antes del cliente oficial y después de sus 20 llamadas, con `contract_version: judge-audio-base64-v1`, 67 señales, bloque `silence_recovery` y umbral 0.5. El pipeline público coincide con el local: `cc5b74284820f8772660f02847509b02e79faa2f45b51a15df72a7ff29626edd`. CORS público devuelve `Access-Control-Allow-Origin: *`.

Se ejecutó exactamente este comando con el cliente oficial sin modificaciones, semilla predeterminada 0 y timeout predeterminado de 30 s:

```bash
python scripts/check_endpoint.py --url https://altur-detector.onrender.com/detect --split val --n 20
```

| Medida del cliente oficial público | Resultado |
| --- | ---: |
| Respuestas válidas | 20/20 |
| Errores HTTP, conexión o contrato | 0 |
| Accuracy | 95.00% (19/20) |
| Balanced accuracy | 95.45% |
| Humanos correctos | 10/11 |
| Sintéticos correctos | 9/9 |
| AUC impreso por el cliente | 0.990 |
| Brier impreso por el cliente | 0.052 |
| Tiempo medio HTTP impreso | 5.318 s |
| Tiempo máximo HTTP impreso | 8.351 s |

El único error de clasificación fue un falso positivo humano. Los 20 veredictos coinciden con la misma muestra ejecutada en local. La muestra pública de 20 llamadas y la validación completa de 71 tienen denominadores distintos; 95.00% en esta muestra no reemplaza el 97.18% de val completa. El cliente imprime floats con tres decimales; balanced accuracy exacta se calculó de sus etiquetas y veredictos: `(10/11 + 9/9) / 2`. AUC, Brier y tiempos medios/máximos se conservan a la precisión impresa.

### Llamada más larga contra la URL pública

Se inspeccionaron las cabeceras de los **353 WAV** y se seleccionó el máximo de `nframes / framerate`, **273.9 segundos**, estéreo PCM16 a 8 kHz y 8,764,844 bytes. El manifest indica 273 s para esa misma llamada; la diferencia de 0.9 s es de metadatos y se reporta la duración real del WAV. Se envió íntegra mediante `post_call()` del cliente oficial al mismo `/detect`, con timeout 30 s e instancia despierta mediante `/health`.

| Medida externa | Segundos |
| --- | ---: |
| Tiempo total, incluida preparación local | 6.699246 |
| Petición y respuesta HTTPS | 6.651573 |
| Lectura, base64, JSON y validación local | 0.047673 |
| Margen frente a 30 s | 23.300754 |

Respuesta válida y dentro del límite con margen. El tiempo total incluye lectura del WAV, base64, JSON, subida, red, procesamiento del servidor y validación de la respuesta. El tiempo HTTPS incluye red y procesamiento; no se presenta como tiempo aislado de inferencia. Es una medición con instancia despierta, no una medición de arranque en frío.

Registro agregado y procedencia: `reports/judge_public_contract.json`. La salida completa sin editar del cliente se entregó en la conversación; su SHA-256 queda en el registro. Los audios y las predicciones individuales permanecen fuera de Git.

### Umbral para balanced accuracy

Se recalcularon las probabilidades de las 71 llamadas de val desde sus WAV y turnos VAD, sin reentrenar el modelo de 67 señales ni modificar el VAD. Se evaluaron todos los cortes de decisión distintos, además de 0, 0.5 y 1; en empates se conserva el umbral más cercano a 0.5. Positivo = sintético; regla `P(synthetic) > threshold`.

| Estado | Umbral | Balanced accuracy | Accuracy | Humanos correctos | Sintéticos correctos | AUC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Antes | 0.5 | 97.30% | 97.18% (69/71) | 35/37 | 34/34 | 0.990461 | 0.030627 |
| Después de buscar el óptimo | 0.5 | 97.30% | 97.18% (69/71) | 35/37 | 34/34 | 0.990461 | 0.030627 |

**No se cambia el umbral:** 0.5 ya alcanza el máximo de balanced accuracy. Evidencia reproducible: `python tune_threshold.py`, `reports/threshold.json` y `artifacts/decision_policy.json`. El SHA-256 del modelo se conserva: `3e2de98c6a780225e2a387ffd2fc3c35cba2132a873fc54b2fecc615d6bbb384`. Buscar el umbral sobre val es selección sobre ese conjunto, no validación independiente.

Cliente oficial conservado sin cambios en `scripts/check_endpoint.py`, fuente `alturio/hackmty26` commit `429adf76b15d1bd18e26b50f371ca4f13b0585c0`, SHA-256 `593f78ceb80017e791f0f8d552ca6a7b3b6763c11beedfa1f68ab1a55c363364`.

## Estado y métricas históricas

Fase 1 aprobada. Fase 2 **CERRADA**: las 71 llamadas evaluadas contra Render reproducen el baseline local.
Docker está aplazado por indicación del usuario. Fase 3: Completada: bloques y combinaciones medidos por HTTP. Selección: silence_recovery.
Prueba de aceleración: completada, con fragilidad documentada abajo.

| Variante | Accuracy | AUC | EER | Brier |
| --- | ---: | ---: | ---: | ---: |
| Referencia: turnos oficiales | 95.77% (68/71) | 0.980922 | 0.054054 | 0.035343 |
| VAD original sin ajustar | 91.55% (65/71) | 0.959459 | 0.117647 | 0.080870 |
| Fase 1: VAD seleccionado en train | 92.96% (66/71) | 0.984897 | 0.054054 | 0.045284 |
| Fase 2: HTTP local, baseline | 92.96% (66/71) | 0.984897 | 0.054054 | 0.045284 |
| Fase 2: HTTPS público, baseline | 92.96% (66/71) | 0.984897 | 0.054054 | 0.045284 |
| Fase 3: modelo seleccionado por HTTP local | 97.18% (69/71) | 0.990461 | 0.054054 | 0.030627 |

Positivo = sintético. EER interpolado en el cruce FAR/FRR. `confidence` expresa la probabilidad de la clase devuelta; AUC y Brier utilizan `P(synthetic)`. No se ha aplicado calibración adicional. Esta tabla histórica recoge corridas de 71 llamadas: el modelo nuevo por HTTP local y el baseline por HTTPS público. El chequeo actual del modelo nuevo con 20 llamadas públicas está documentado en el apartado del contrato del juez.

## Datos y separación

353 llamadas: train 282 (113 humanas, 169 sintéticas), val 71 (37 humanas, 34 sintéticas). Los 353 WAV son estéreo, 8 kHz y PCM16. Se respeta el split oficial por hablante; ningún ajuste del clasificador usa val. Val sí se utiliza para seleccionar los bloques de Fase 3: sus ganancias son exploratorias, sin un segundo conjunto independiente. Cada acierto en val equivale a 1.41 puntos porcentuales.

README oficial leído antes de empezar: `alturio/hackmty26`, commit `26b519598a1520cf6306d78902ef5047ae670aa4`, release `v1.0`. El código original queda en `baseline_original/`. Sus 58 predictores más tres columnas de metadatos suman las 61 columnas originales. Se corrigió el ajuste final train+val del script recibido para entrenar únicamente con train.

El dataset sigue excluido del repositorio. La petición posterior del usuario autorizó enviar las 71 llamadas a su endpoint HTTPS de Render para evaluarlo. El servicio procesa cada WAV en memoria y no lo almacena. Todo entrenamiento, extracción experimental y estrés se realiza en el entorno local.

## Fase 1: VAD congelado

Malla de **540 configuraciones**, seleccionada únicamente por correlación con turnos de referencia en train. El clasificador de producción usa turnos del WAV, sin JSON de referencia. Los cinco parámetros elegidos son:

```json
{
  "frame_ms": 30,
  "thresh_db": -46.0,
  "min_speech": 0.1,
  "min_sil": 0.2,
  "noise_margin": 18.0
}
```

| Feature | Pearson train | Pearson val |
| --- | ---: | ---: |
| `lat_med` | 0.947717 | 0.947633 |
| `lat_mean` | 0.922605 | 0.925372 |
| `barge_in_rate` | 0.894513 | 0.875019 |
| `overlap_ratio` | 0.887432 | 0.842966 |
| `n_caller` | 0.792691 | 0.752486 |

Puerta: r de `lat_med` > 0.9 y accuracy 92.96%, dentro de 3 puntos del 95.8% de referencia. Se conservan la malla completa y correlaciones de las 58 variables en `reports/vad_grid_train.csv`, `reports/vad_correlations_train.csv` y `reports/vad_correlations_val.csv`.

El diagnóstico de bleed encontró 0 llamadas de train con correlación instantánea absoluta > 0.6 y coeficiente lineal absoluto > 0.003 simultáneamente. No se restaron canales; esto no descarta bleed con retardo o no lineal.

## Fase 2: evaluación pública completa

URL: [altur-detector.onrender.com](https://altur-detector.onrender.com). Se llamó primero a `/health`, seguido de 71 peticiones secuenciales a `/detect`. Registro: `reports/phase2_public_http.json`.

- Inicio UTC: `2026-09-12T05:08:42.788444+00:00`.
- Fin UTC: `2026-09-12T05:09:30.321018+00:00`.
- Peticiones satisfactorias: 71; errores: 0; reintentos de inferencia: 0; abstenciones: 0.
- Diferencia máxima frente a probabilidades offline: `1.3270634591222574e-16`.
- Modelo público SHA-256: `135723cf56e554b006b224430d2a5dcb2d6f2950d7e86f75d28f70615ff8ea69`.
- Pipeline público SHA-256: `fcaac6c00f5900c35a29df8393f57455dee3111ae7c0abdc7bf7973aa1dca470`.

| Medición externa | Milisegundos |
| --- | ---: |
| `/detect` p50 | 586.69 |
| `/detect` p95 | 839.89 |
| `/health` previo, separado de percentiles | 5591.42 |

Latencias medidas desde este entorno hacia la URL pública, con servidor despierto y conexión del cliente reutilizada. Incluyen serialización JSON, subida, red, inferencia y respuesta; excluyen lectura del WAV y base64. Son una corrida secuencial de 71 llamadas, no una prueba de concurrencia ni disponibilidad prolongada. El cliente respeta el proxy de red del entorno para HTTPS; loopback no usa proxy.

## Fase 3: ablación por bloque

Mismo VAD y mismos hiperparámetros HGB: 300 iteraciones, learning rate 0.06, 15 hojas, L2=1, semilla 0. Se entrenó exclusivamente con 282 llamadas de train. Cada fila recorrió las 71 llamadas de val por el mismo `POST /detect` local y reprodujo las probabilidades offline con error < 1e-12.

Regla de inclusión fijada: mejora estricta en accuracy frente a 66/71. Entre ganadores, elegir primero mayor accuracy y menos features en empate; añadir otro bloque solo si vuelve a aumentar accuracy. AUC/Brier se reportan por separado. No se ajustaron parámetros, umbral ni bins con los resultados de val.

| Baseline + bloque | Features nuevas | Accuracy | Δ pp vs baseline | AUC | Brier | Decisión |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Recuperación tras interrupción | +10 | 92.96% (66/71) | +0.00 | 0.985692 | 0.043488 | Fuera: accuracy sin mejora |
| Recuperación tras silencio | +9 | 97.18% (69/71) | +4.23 | 0.990461 | 0.030627 | INCLUIDO |
| Consistencia | +6 | 95.77% (68/71) | +2.82 | 0.981717 | 0.035228 | Mejora solo; no añade aciertos al combinar |
| Deriva temporal | +2 | 92.96% (66/71) | +0.00 | 0.984102 | 0.048571 | Fuera: accuracy sin mejora |
| Autocorrelación | +2 | 92.96% (66/71) | +0.00 | 0.986486 | 0.045053 | Fuera: accuracy sin mejora |
| Recuperación tras silencio + Consistencia | +15 | 97.18% (69/71) | +4.23 | 0.990461 | 0.030967 | Fuera: no mejora al mejor bloque |

- **Interrupción:** primera entrada del agente dentro de cada turno del llamante; fracción que se calla en 0.5/1 s y antes de que termine el agente. Reanudación antes del siguiente turno del agente, limitada a 12 s después del agente y al fin de habla observado. Incluye tasa de reanudación, espera desde el cese y desde el final del agente, y duración reanudada. Ventanas sin observación se marcan NaN. El baseline ya incluía estadísticas `yield_*` del tiempo hasta callarse.
- **Silencio:** pausas entre turnos del agente > 3 s. Para el primer turno que comienza dentro de la pausa se mide espera, duración hasta el regreso del agente y duración completa. Se agregan mediana/p90/desviación de espera y duración, mediana de duración completa, espera dividida por duración del hueco y fracción ocupada. Son nueve features. Son proxies temporales, sin inferir intención o contenido del silencio.
- **Consistencia:** se reutilizan std/IQR/CV de latencias, std de duraciones y CV del llamante ya existentes. Se añaden entropía de latencias, IQR y entropía de ambas duraciones, y CV de duración del agente. Entropía Shannon normalizada con bins fijos en `behavior_features.py`.
- **Deriva:** pendiente OLS de latencia frente al índice original del turno del llamante y R²; al menos tres observaciones.
- **Autocorrelación:** Pearson de la serie consigo misma a retardos 1 y 2; al menos tres pares. Series constantes o insuficientes dan NaN.

Las features se recomputan desde los turnos VAD en memoria para entrenamiento y evaluación. Se eliminó la lectura de floats redondeados desde CSV como entrada del experimento: una diferencia de un ULP puede cruzar un umbral de árbol. La primera combinación presentó una discrepancia HTTP/offline y fue rechazada; se repitió toda la ablación tras corregir la fuente de floats. La tabla contiene únicamente la repetición verificada.

Modelo seleccionado: `silence_recovery`; SHA-256 `3e2de98c6a780225e2a387ffd2fc3c35cba2132a873fc54b2fecc615d6bbb384`. `artifacts/model.joblib` contiene el seleccionado; `artifacts/baseline_model.joblib` conserva el baseline público. El endpoint calcula solo los bloques declarados por el artefacto. Los bloques descartados quedan disponibles para reproducir la ablación y no se ejecutan en la inferencia del modelo seleccionado.

Verificación final: 17 pruebas de bordes aprobadas. `reports/phase3_final_clean_http.json` registra las 71 peticiones con servidor en un entorno Python aislado que contiene solo dependencias runtime. Diez fixtures analíticos comprueban interrupciones, censura, relleno, tendencia, autocorrelación y desplazamiento de latencias.

## Robustez: sintético más rápido

Se congelaron ambos modelos antes de la prueba. Se modificaron exclusivamente las 34 llamadas sintéticas de val; se comprobó que las probabilidades de las 37 humanas permanecieran idénticas. No se reentrenó ni eligieron bloques usando estos escenarios.

**Solo latencias:** resta 1.0/1.5 s a cada latencia original válida, conserva emparejamientos y observaciones, permite valores negativos y recalcula estadísticas, porcentajes bajo/sobre umbral, CV y las features nuevas dependientes de esa serie. Mantiene solapamientos, interrupciones y rellenos de silencio. Por ello, el 97.18% sin caída del modelo nuevo en esta intervención parcial no demuestra robustez del motor: deja intactas sus nuevas features de silencio.

**Todos los turnos:** prueba adicional que adelanta todos los intervalos VAD del llamante sintético, conserva el agente, recorta tiempos a cero y recalcula todas las features, incluidos solapamientos y rellenos. Puede cambiar el emparejamiento o filtrado de latencias. Resuelve el sesgo optimista de congelar esas señales al acelerar solo `lat_*`.

| Modelo | Intervención | Adelanto | Accuracy | Caída pp | Recall sintético |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Solo latencias | −0 s | 92.96% (66/71) | 0.00 | 97.06% |
| Baseline | Solo latencias | −1 s | 78.87% (56/71) | 14.08 | 67.65% |
| Baseline | Solo latencias | −1.5 s | 76.06% (54/71) | 16.90 | 61.76% |
| Baseline | Todos los turnos del llamante | −0 s | 92.96% (66/71) | 0.00 | 97.06% |
| Baseline | Todos los turnos del llamante | −1 s | 81.69% (58/71) | 11.27 | 73.53% |
| Baseline | Todos los turnos del llamante | −1.5 s | 77.46% (55/71) | 15.49 | 64.71% |
| Baseline + silencio | Solo latencias | −0 s | 97.18% (69/71) | 0.00 | 100.00% |
| Baseline + silencio | Solo latencias | −1 s | 97.18% (69/71) | 0.00 | 100.00% |
| Baseline + silencio | Solo latencias | −1.5 s | 97.18% (69/71) | 0.00 | 100.00% |
| Baseline + silencio | Todos los turnos del llamante | −0 s | 97.18% (69/71) | 0.00 | 100.00% |
| Baseline + silencio | Todos los turnos del llamante | −1 s | 71.83% (51/71) | 25.35 | 47.06% |
| Baseline + silencio | Todos los turnos del llamante | −1.5 s | 63.38% (45/71) | 33.80 | 29.41% |

Son intervenciones sobre features/turnos, no WAVs de un motor nuevo ni una medición HTTP de audio acelerado. El desplazamiento uniforme de todos los turnos también es una simplificación. Los números miden sensibilidad, no rendimiento esperado del set oculto.

**Conclusión para el pitch:** El baseline depende fuertemente de la latencia. El bloque de silencio mejora val, pero también depende del tiempo: al desplazar todos los turnos 1.5 s, el seleccionado cae a 63.38% y el baseline a 77.46%. La ganancia en val no demuestra generalización frente a un motor más rápido. Harían falta aumentación temporal y señales independientes, evaluadas antes de incluirse; no se presentan como implementadas.

## Reproducción y alcance

```bash
make all PUBLIC_URL=https://altur-detector.onrender.com
```

Instala dependencias, verifica/descarga los datos fijados, ejecuta Fase 1, verifica baseline local y público, corre ablaciones por HTTP y estrés. Las puertas se ejecutan secuencialmente. La comprobación pública requiere que esa URL sirva el baseline cuya métrica se compara; falla si el despliegue cambió. Para repetir Fase 3 después del cierre público registrado: `make phase3 stress`.

Artefactos de evidencia: `reports/phase2_public_http.json`, `reports/phase3.json`, `reports/phase3_final_clean_http.json`, `reports/phase3_final_edges.json`, `reports/stress_latency.json`. Predicciones por llamada y cachés permanecen locales. No se ha medido semántica, acústica, clips parciales, ruido/ganancia ni calibración adicional. La demo sigue pendiente.
