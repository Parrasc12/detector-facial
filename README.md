# Detector Facial Cianótipo

Aplicação de análise de expressões e reconhecimento facial executada no navegador. A câmera só é ativada depois de uma ação explícita. A interface usa grafite e prata, com azul suave em pequenos indicadores e rosa apenas em detalhes.

## Requisitos e execução

- Python 3 para servir os arquivos estáticos; nenhum pacote Python adicional é necessário.
- Navegador moderno com suporte a módulos ES, WebAssembly, MediaPipe Tasks Vision, câmera e IndexedDB.
- Conexão à internet para carregar a biblioteca e os modelos de IA do jsDelivr e do Google Cloud Storage.

Abra um terminal nesta pasta e execute:

```powershell
python -m http.server 8000


Depois abra `http://localhost:8000` ou `http://127.0.0.1:8000` no Brave, Chrome ou Edge. A câmera exige `localhost` ou HTTPS. Autorize o acesso somente se quiser usar os recursos ao vivo.

Não há etapa de build nem instalação de dependências locais. O MediaPipe Face Landmarker é carregado ao abrir a página; face-api.js e os modelos de reconhecimento só são carregados quando o reconhecimento ou o cadastro facial é solicitado.

## Detecção e reconhecimento

- A detecção básica usa MediaPipe Face Landmarker, mostra a malha facial, estima expressões visuais e suporta até quatro rostos.
- O reconhecimento de identidade é separado da detecção: usa descritores faciais de 128 dimensões gerados por face-api.js e comparação local por distância euclidiana.
- O reconhecimento é opcional e pode ser desligado sem interromper a detecção básica. As correspondências são aproximadas, não comprovam identidade; o índice de proximidade não é uma probabilidade.
- Cadastros por foto exigem nome, consentimento e ao menos uma imagem com exatamente um rosto. O cadastro ao vivo exige câmera ativa e um único rosto enquadrado. Não cadastre pessoas sem autorização.
- Economia, Equilibrado e Qualidade ajustam resolução e frequência-alvo da análise. O desempenho real depende do navegador e do computador; não há garantia de FPS.

## Privacidade e armazenamento

Fotos e quadros são processados localmente e não são enviados pelo aplicativo. Os arquivos de imagem originais não são guardados. O cadastro persiste no IndexedDB do navegador e contém nome, data do cadastro e descritores faciais — dados biométricos sensíveis. Os registros ficam limitados a este navegador/origem e podem ser removidos individualmente na interface.

Obtenha autorização explícita antes de cadastrar alguém. Excluir um cadastro remove seus descritores do IndexedDB deste navegador. A aplicação avisa quando o armazenamento persistente não pôde ser confirmado; nesse caso, o navegador pode apagar os dados locais. Faça a gestão dos dados de acordo com as regras de privacidade aplicáveis.

O navegador baixa código e modelos de terceiros para executar a análise, o que requer internet. O aplicativo não envia imagens nem descritores a esses serviços.

## Limitações

- O reconhecimento pode falhar com iluminação, pose, oclusão ou imagem de baixa qualidade e pode produzir falsos positivos ou falsos negativos.
- Cada foto de cadastro deve conter exatamente um rosto detectável. O cadastro ao vivo também exige um único rosto por vez.
- A resolução disponível e a execução dependem do navegador, dispositivo, permissões e capacidade de armazenamento.
- As expressões são estimativas visuais não clínicas. O reconhecimento não deve ser usado como autenticação de segurança.
