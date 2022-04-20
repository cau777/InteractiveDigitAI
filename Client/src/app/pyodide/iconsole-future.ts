export interface IConsoleFuture extends Promise<any> {
    syntax_check: "incomplete" | "complete" | "syntax-error";
    formatted_error: string;
}
