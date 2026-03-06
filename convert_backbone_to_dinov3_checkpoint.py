"""
Convierte el backbone preentrenado de DINOv3 (.pth) al formato de checkpoint
completo que espera init_fsdp_model_from_checkpoint() en ssl_meta_arch.py.

Uso:
    python convert_backbone_to_dinov3_checkpoint.py \
        --input  /content/dinov3_lora/pesos/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
        --output /content/dinov3_lora/pesos/dinov3_vits16_teacher_ckpt.pth
"""

import argparse
import torch


def convert(input_path: str, output_path: str):
    print(f"Cargando backbone desde: {input_path}")
    raw = torch.load(input_path, map_location="cpu")

    # ---------- 1. Extraer el state_dict del backbone ----------
    # El backbone de DINOv3 puede venir directamente como state_dict,
    # o dentro de una key 'model', 'teacher', 'student', etc.
    if isinstance(raw, dict):
        # Intentar keys comunes en orden de preferencia
        for key in ("model", "teacher", "student", "state_dict"):
            if key in raw:
                state_dict = raw[key]
                print(f"  → Extraído desde key '{key}'")
                break
        else:
            # No encontró ninguna key conocida: asumir que ES el state_dict
            state_dict = raw
            print("  → Usando el dict directamente como state_dict")
    else:
        raise ValueError(f"Formato inesperado: {type(raw)}")

    # ---------- 2. Limpiar prefijos indeseados ----------
    # A veces los pesos vienen con prefijos como 'module.', 'backbone.', 'encoder.'
    clean = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("module.", "backbone.", "encoder."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        clean[new_k] = v

    # ---------- 3. Verificar que las keys parecen correctas ----------
    sample_keys = list(clean.keys())[:8]
    print(f"\nPrimeras keys del backbone limpio:")
    for k in sample_keys:
        print(f"  {k}")

    # ---------- 4. Prefixar con 'backbone.' ----------
    # init_fsdp_model_from_checkpoint carga sobre nn.ModuleDict({"backbone": ..., "dino_head": ...})
    # por eso las keys deben empezar con 'backbone.'
    teacher_state_dict = {f"backbone.{k}": v for k, v in clean.items()}

    print(f"\nEjemplo de keys finales con prefijo 'backbone.':")
    for k in list(teacher_state_dict.keys())[:5]:
        print(f"  {k}")

    # ---------- 5. Guardar en formato teacher ----------
    checkpoint = {"teacher": teacher_state_dict}
    torch.save(checkpoint, output_path)
    print(f"\nCheckpoint guardado en: {output_path}")
    print(f"   Total de keys: {len(teacher_state_dict)}")
    print("\nAhora usa este checkpoint en tu comando de entrenamiento con:")
    print(f"   student.resume_from_teacher_chkpt={output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Ruta al .pth del backbone preentrenado")
    parser.add_argument("--output", required=True, help="Ruta donde guardar el checkpoint convertido")
    args = parser.parse_args()
    convert(args.input, args.output)
