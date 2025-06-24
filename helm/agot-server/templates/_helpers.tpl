{{- define "agot-server.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{- define "agot-server.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "agot-server.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}
