# CourseLangGraph

[![](https://dcbadge.vercel.app/api/server/n8w5qE4xyA)](https://discord.gg/n8w5qE4xyA)

Requirements:
- `python >= 3.11`
- `data.db` is required
- `mamba` or `conda` installed
- `ollama` installed ([Github](https://github.com/ollama/ollama))


## Installation
```sh
mamba env create -f environment.yml
```

Add `MODEL` in `.env` to specify your model file

```
MODEL="your_modal_name"
```

> [!IMPORTANT]
> The environment file use cpu version of pytorch, if you have GPUs, install GPU version on your own.

If you want to monitor in LangSmith, add config in the `.env`

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="api-key"
```

## Run
```sh
python main.py
```

### Run with streamlit
```sh
streamlit run streamlit.py
```

### Run with fastapi
```sh
python app.py
```

## Frontend Repository

**[CLick me](https://github.com/NCCUCourseScheduling/CourseLangGraph-frontend)**

## Final Report
Objective, System Architecture, Research Methods, Results, Future Outlook : 
[report](https://docs.google.com/document/d/1CkelC_x8B_QnVHEiIZisG1d8BJwXoYg02lqaqgbQlFY/edit?usp=sharing) 

## Demo
![image](https://github.com/NCCUCourseScheduling/CourseLangGraph/assets/74034659/ac8269dc-1765-48f6-a1bd-fcd8fc09a383)

## System Structure
![image](https://github.com/NCCUCourseScheduling/CourseLangGraph/assets/74034659/f23cecc2-b9c4-42c3-a684-f33c799d33e7)

![image](https://github.com/NCCUCourseScheduling/CourseLangGraph/assets/74034659/eb04a8a1-557e-428b-80d0-fcb8497b5562)


