// noinspection JSFileReferences
/// <reference lib="webworker" />

import {PyodideOutputMessage, PyodideWorkerMessage} from "./pyodide-worker-messages";
import {PyodideWorkerLogic} from "./pyodide/pyodide-worker-logic";

importScripts("https://cdn.jsdelivr.net/pyodide/v0.20.0/full/pyodide.js");

function stdCallback(content: string, isError: boolean) {
    if (!content) return;
    let message: PyodideOutputMessage = {
        action: "output",
        content: content,
        isError: isError
    };
    postMessage(message);
}

let worker: PyodideWorkerLogic  = new PyodideWorkerLogic();

function selectAction(data: PyodideWorkerMessage): Promise<unknown | undefined> {
    switch (data.action) {
        case "load":
            return worker.load(data.libs, data.libsArchives, stdCallback);
        case "select":
            return worker.select(data.code, data.params);
        case "run":
            return worker.run(data.expression, data.params, stdCallback);
    }
    throw new RangeError(data.action);
}

addEventListener("message", ({data}: { data: PyodideWorkerMessage }) => {
    let action = selectAction(data);
    
    action.then(r => {
        postMessage({action: "result", content: r});
    }, e => {
        throw e;
    });
});
