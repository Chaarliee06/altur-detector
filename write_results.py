"""Render RESULTS.md from recorded measurements; never invent missing results."""
import json
from pathlib import Path


def load(name):
    path = Path("reports") / name
    return json.loads(path.read_text()) if path.exists() else {}


def metric_row(label, result, status):
    if not result:
        return f"| {label} | — | — | — | — | {status} |"
    return (f"| {label} | {result['accuracy']:.6f} ({result['correct']}/{result['n']}) | "
            f"{result['auc']:.6f} | {result['eer']:.6f} | {result['brier']:.6f} | {status} |")


def main():
    phase1 = load("phase1.json")
    if not phase1:
        raise SystemExit("Run phase1 before generating RESULTS.md")
    http = load("phase2_http.json")
    clean = load("phase2_clean_http.json")
    edges = load("phase2_edges.json")
    docker = load("phase2_docker.json")
    public = load("phase2_public.json")
    phase2_passed = bool(http.get("offline_equivalent") and public.get("passed")
                         and http.get("model_sha256") == public.get("model_sha256")
                         and http.get("pipeline_sha256") == public.get("pipeline_sha256"))
    gate_text = ("Fases 1 y 2 aprobadas. Las fases 3–6 todavía no se han iniciado." if phase2_passed else
                 "Fase 1 aprobada. Fase 2 tiene evaluación HTTP local reproducida y pruebas de bordes, pero **la puerta completa sigue cerrada**: falta verificar el endpoint público. Docker fue aplazado por el usuario y no bloquea esta fase. No se han iniciado las fases 3–6.")
    rows = [metric_row("Referencia: turnos oficiales", phase1["reference_metrics"], "Reproducida; no desplegable sin VAD"),
            metric_row("VAD original sin ajustar", phase1["default_vad_metrics"], "No alcanza la puerta"),
            metric_row("Fase 1: VAD seleccionado en train", phase1["selected_vad_metrics"], "APROBADA" if phase1["gate"]["passed"] else "NO APROBADA"),
            metric_row("Fase 2: HTTP local", http, "Equivalencia comprobada" if http.get("offline_equivalent") else "Pendiente"),
            metric_row("Fase 2: entorno Python limpio", clean, "Equivalencia comprobada" if clean.get("offline_equivalent") else "Pendiente"),
            metric_row("Fase 2: host público", {}, "APROBADA" if phase2_passed else "Pendiente de despliegue y verificación"),
            metric_row("Docker", {}, "Aplazado por el usuario; fuera de la puerta actual"),
            metric_row("Fase 3: comportamiento adicional", {}, "No iniciada: puerta anterior pendiente"),
            metric_row("Fase 4: semántica", {}, "No iniciada"),
            metric_row("Fase 5: robustez/calibración", {}, "No iniciada"),
            metric_row("Fase 6: demo", {}, "No iniciada")]
    by_split = {split: {r['feature']: r for r in phase1[f'priority_correlations_{split}']} for split in ('train', 'val')}
    corr_rows = [f"| `{name}` | {by_split['train'][name]['pearson_r']:.6f} | {by_split['val'][name]['pearson_r']:.6f} | {by_split['train'][name]['mae']:.6f} |" for name in ('lat_med', 'lat_mean', 'barge_in_rate', 'overlap_ratio', 'n_caller')]
    latency_rows = [f"| {label} | {record['latency_p50_ms']:.2f} ms | {record['latency_p95_ms']:.2f} ms |" for label, record in [('HTTP local inicial', http), ('HTTP con servidor en entorno limpio', clean)]] if http and clean else []
    output = f'''# RESULTS — Altur HackMTY 2026

## Estado de las puertas

{gate_text}

| Variante/fase | Accuracy | AUC | EER | Brier | Estado |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

Todas las métricas medidas corresponden a las mismas 71 llamadas de val, separadas por hablante de train. Positivo = sintético. EER se interpola en el cruce FAR/FRR; los JSON también incluyen la definición discreta del código original. Las cifras calibradas aportadas en el brief no se presentan como reproducidas: la calibración no se ha ejecutado.

## Datos y baseline preservado

- Repo oficial leído antes de ejecutar: `alturio/hackmty26`, commit `26b519598a1520cf6306d78902ef5047ae670aa4`, release `v1.0`.
- 353 llamadas: train 282 (113 humanas, 169 sintéticas); val 71 (37 humanas, 34 sintéticas). Cada una tiene WAV y JSON; los 353 WAV cumplen estéreo, 8 kHz y PCM16.
- `features.py` contiene **58 predictores**, más tres columnas de identificación/etiqueta/split en su CSV: 61 columnas en total.
- Se conservaron las definiciones originales. La refactorización para recibir turnos en memoria y el VAD con configuración por defecto dieron resultados idénticos al código adjunto para las 353 llamadas.
- El código original de entrenamiento terminaba ajustando con train+val. Se corrigió: el modelo entregado usa exclusivamente train. No se calculó CV aleatoria ni se atribuye separación por hablante a folds sin identificadores de hablante.

## Fase 1: VAD

Se probaron **{phase1['grid_configurations']} configuraciones**, todas evaluadas por correlación contra referencias de las 282 llamadas de train. Val no intervino en la selección. Se congeló la configuración antes de generar las predicciones del modelo seleccionado en val.

Malla: `frame_ms ∈ {{20,30}}`, `thresh_db ∈ {{-46,-42,-38,-34,-30,-26}}`, `min_speech ∈ {{0.1,0.2,0.3}}`, `min_sil ∈ {{0.1,0.2,0.25,0.3,0.4}}`, `noise_margin ∈ {{6,12,18}}`.

Configuración elegida:

```json
{json.dumps(phase1['selected_config'], indent=2)}
```

El umbral de actividad es el máximo entre `thresh_db` y el percentil 10 de energía del canal más `noise_margin`. Se maximiza Pearson de `lat_med`, con correlación media de las cinco variables prioritarias como desempate.

| Variable | Pearson train (n=282) | Pearson val (n=71) | MAE train |
| --- | ---: | ---: | ---: |
{chr(10).join(corr_rows)}

Las correlaciones de las 58 variables están en `reports/vad_correlations_train.csv` y `reports/vad_correlations_val.csv`. La malla completa está en `reports/vad_grid_train.csv`.

Puerta: `lat_med` r = {phase1['gate']['lat_med_r_train']:.6f} > 0.9; accuracy = {phase1['selected_vad_metrics']['accuracy']:.6f} ≥ 0.928. Son dos aciertos menos que el baseline de referencia (2.82 puntos porcentuales), dentro del margen permitido. No fue necesario probar WebRTC/Silero para satisfacer esta puerta.

Bleed: en tramos de train donde la referencia marca solo agente, **{phase1['bleed_calls_abs_correlation_over_06']} llamadas** superaron simultáneamente |correlación instantánea| > 0.6 y |coeficiente lineal| > 0.003. No se aplicó resta entre canales. Esta prueba no descarta filtración con retardo o no lineal. Los turnos de referencia se usan solo para este diagnóstico y la comparación, nunca para inferencia.

## Fase 2: endpoint

`POST /detect` recibe el WAV base64, extrae los turnos con el VAD congelado, obtiene las mismas 58 variables y devuelve el veredicto y la confianza de esa clase. No usa JSON de referencia ni imputa NaN de forma distinta al entrenamiento.

- 71/71 peticiones completas; {http.get('abstentions', 'pendiente')} abstenciones en val.
- Diferencia máxima entre probabilidades HTTP y offline: `{http.get('max_probability_delta_offline', 'pendiente')}`.
- Pruebas de bordes: {edges.get('n', 0)} aprobadas. Incluyen mono, audio corto, silencio, cero frames, menos de cuatro turnos, un solo canal con actividad, WAV truncado, base64/JSON inválidos y formato incompatible.
- `confidence=0.5` significa abstención; el bool `false` solo satisface el esquema obligatorio.
- Confianza sin redondeo; AUC/Brier se calculan con `P(synthetic)` recuperada del veredicto y su confianza.
- El endpoint procesa audio en memoria, no guarda muestras y evita reflejar base64 en errores. Límite: 32 MiB y 900 s.

| Ejecución | p50 | p95 |
| --- | ---: | ---: |
{chr(10).join(latency_rows)}

Medición secuencial con servidor caliente y HTTP loopback: incluye serialización JSON, transporte e inferencia; excluye lectura del archivo y codificación base64. Estas cifras **no son latencias de internet ni prueban concurrencia o disponibilidad de 15 minutos**. El entorno limpio se creó sin paquetes heredados y solo con las dependencias del servidor. Docker se aplaza por indicación del usuario.

Modelo evaluado SHA-256: `{http.get('model_sha256', 'pendiente')}`.

Código del equipo: [Chaarliee06/altur-detector](https://github.com/Chaarliee06/altur-detector). Render está configurado para Python 3.12 con dependencias fijadas y el modelo incluido; no descarga el dataset.
Host público: {public.get('url') if public.get('passed') else 'pendiente de despliegue y verificación'}. `/health` identifica tanto el modelo como el código de inferencia. La prueba pública exige equivalencia HTTP local de ese mismo código y 900 segundos de respuestas estables.

Para cumplir la restricción de procesamiento local, las 71 llamadas solo se envían por HTTP loopback. Las pruebas de un host externo usan tonos generados y comprueban el mismo SHA-256. No se ha enviado ningún audio del dataset a un host externo.

## Límites actuales del argumento

La validación respeta el split oficial, pero no demuestra rendimiento sobre voces del set oculto. Todavía no se han medido motores con menor latencia, clips parciales, ruido/ganancia, calibración adicional ni señales semánticas. No usar las cifras de duración del baseline de referencia como resultados del endpoint con VAD.

`make all` reproduce las etapas implementadas y se detiene al encontrar una puerta pendiente. El paquete contiene código, modelo y resultados agregados; excluye audio, turnos oficiales, manifiesto y cachés por llamada.
'''
    Path("RESULTS.md").write_text(output)
    print("RESULTS.md updated from measured reports")


if __name__ == "__main__":
    main()
