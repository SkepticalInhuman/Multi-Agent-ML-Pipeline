import jupyter_client
from queue import Empty

class JupyterSandbox:
    def __init__(self):
        self.km, self.kc = jupyter_client.manager.start_new_kernel(kernel_name='python3')

    def execute_code(self, code: str) -> str:
        msg_id = self.kc.execute(code)

        output = []

        while True:
            try:
                io_msg = self.kc.get_iopub_msg(timeout=60)
            except Empty:
                output.append("ERROR: Timed out waiting for kernel output.")
                break

            # Ignore messages from previous executions
            if io_msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = io_msg["header"]["msg_type"]
            content = io_msg["content"]

            if msg_type == "stream":
                output.append(content["text"])

            elif msg_type == "execute_result":
                if "text/plain" in content["data"]:
                    output.append(content["data"]["text/plain"])

            elif msg_type == "display_data":
                if "text/plain" in content["data"]:
                    output.append(content["data"]["text/plain"])

            elif msg_type == "error":
                output.append(
                    f"{content['ename']}: {content['evalue']}\n"
                    + "\n".join(content["traceback"])
                )

            # The important part: execution has finished
            elif (
                msg_type == "status"
                and content["execution_state"] == "idle"
            ):
                break

        full_output = "\n".join(output)

        MAX_CHAR_LIMIT = 1000

        if len(full_output) > MAX_CHAR_LIMIT:
            full_output = (
                full_output[:MAX_CHAR_LIMIT]
                + f"\n\n... [TRUNCATED {len(full_output)-MAX_CHAR_LIMIT} CHARACTERS] ..."
            )

        return full_output if full_output else "Code executed successfully."

    def shutdown(self):
        self.km.shutdown_kernel()

sandbox = JupyterSandbox()