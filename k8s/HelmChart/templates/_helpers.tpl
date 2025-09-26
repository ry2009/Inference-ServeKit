{{- define "primerl-servekit.name" -}}
{{- .Chart.Name -}}
{{- end -}}

{{- define "primerl-servekit.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "primerl-servekit.redisHost" -}}
{{- if .Values.redis.enabled -}}
{{- printf "%s-redis" (include "primerl-servekit.fullname" .) -}}
{{- else -}}
{{- .Values.redis.host | default "redis" -}}
{{- end -}}
{{- end -}}
